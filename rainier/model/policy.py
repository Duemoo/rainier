import os
import json
from typing import Union, List, Dict, Tuple
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model.t5 import T5ForConditionalGenerationAndTokenRegression
from utils.utils import logits_to_entropy, mask_pad


class Policy:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 policy_value_sharing: bool,
                 max_input_len: int,
                 max_output_len: int,
                 device,
                 batch_size,
                 kl_coef,
                 ensembling,
                 num_pooling_layers,
                 device_map = None,
                ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)
        if policy_value_sharing:
            self.model = T5ForConditionalGenerationAndTokenRegression.from_pretrained(model_type)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_type)
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model.eval()
        self.model.to(device)
        if device != 'cpu':
            self.model.parallelize(device_map=device_map)

        self.policy_value_sharing = policy_value_sharing
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.device = device

        self.gain, self.bias = None, None
        self.batch_size = batch_size
        self.kl_coef = kl_coef
        self.ensembling = ensembling
        self.num_pooling_layers = num_pooling_layers

        #Update decoder part only
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def sample(self,
               text: List[str],
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:

        tokenized = self.tokenizer.batch_encode_plus(
            text,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.max_input_len)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        response_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_len + 1,
            min_length=3,
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        response_ids = response_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        response_mask = (response_ids != self.model.config.pad_token_id).int()
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with torch.no_grad():
            outputs = self.forward_pass(input_ids, attention_mask, response_ids, response_mask)
        response_logits = outputs['response/logits']
        response_logprobs = outputs['response/logprobs']
        response_entropy = outputs['response/entropy']

        return {
            'query/text': text,
            'query/input_ids': input_ids,
            'query/mask': attention_mask,
            'response/text': response_text,
            'response/input_ids': response_ids,
            'response/mask': response_mask,
            'response/logits': response_logits,
            'response/logprobs': response_logprobs,
            'response/entropy': response_entropy,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                    ):

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        response_logits = outputs.logits # (B, RL, V)
        logprobs = F.log_softmax(response_logits, dim=-1)
        response_logprobs = torch.gather(logprobs, 2, response_input_ids[:, :, None]).squeeze(2) # (B, RL)
        response_entropy = logits_to_entropy(response_logits) # (B, RL)

        return {
            'response/logits': response_logits,
            'response/logprobs': mask_pad(response_logprobs, response_mask),
            'response/entropy': mask_pad(response_entropy, response_mask),
        }

    #Reward utils
    def get_reward(self,
                   questions: List[str],
                   knowledges: List[str],
                   answer: List[str],
                   override_gain = None,
                   override_bias = None,
                   skip_reward = False,
                  ) -> Tuple[List[float], float, int, int]:
        if knowledges is None:
            knowledges = [None for _ in questions]

        assert len(questions) == len(knowledges)

        questions = [a.lower() for a in questions]
        knowledges = [a.lower() if a is not None else None for a in knowledges]
        answers = [a.lower() for a in answer]
        
        # 2D tensor(bs * hidden_dim)
        k_emb = self.get_embedding(knowledges)
        ans_emb = self.get_embedding(answers)

        similarity = (k_emb*ans_emb).sum(dim=-1)   # 1D tensor(bs)

        rewards_raw = similarity.tolist()
        #print(f"rewards_raw: {rewards_raw}")

        #if skip_reward:
        #    return {
        #        'preds': knowledges,
        #    }

        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        rewards_normalized = [gain * x + bias for x in rewards_raw]

        return {
            'preds': knowledges,
            'rewards/raw': rewards_raw,
            'rewards/normalized': rewards_normalized,
        }

    def kl_penalize_reward(self, results):
        logprobs = results['response/logprobs']
        ref_logprobs = results['response/ref_logprobs']
        mask = results['response/mask']
        normalized_rewards = results['rewards/normalized']

        kl = logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl
        RL = logprobs.size(1)
        flattened_rewards = torch.tensor([
            [0.] * (l-1) + [r] + [0.] * (RL-l)
            for r, l in zip(normalized_rewards, torch.sum(mask, dim=1).tolist())
        ], device=logprobs.device) # (B, RL)
        penalized_rewards = flattened_rewards - kl_penalty
        # TODO: This is slightly different from the paper

        results['rewards/kl'] = kl
        results['rewards/kl_penalty'] = kl_penalty
        results['rewards/penalized'] = penalized_rewards
    
    def get_reward_ensemble(self,
                            questions: List[str],
                            knowledgess: List[List[str]],
                            choicess: List[List[str]],
                            answer_ixs: List[int],
                            override_gain = None,
                            override_bias = None,
                           ) -> Tuple[List[float], float, int, int]:

        answer_sims = []

        for knowledges in knowledgess:
            results = self.get_reward(questions, knowledges, answer, override_gain, override_bias, skip_reward=False)

        return {
            'preds': results['preds'],
            'rewards': results['rewards/normalized']
        }

    def write_reward_norm(self, reward_dir):
        reward_dict = {
            'gain': self.gain,
            'bias': self.bias,
        }
        with open(os.path.join(reward_dir, 'reward_normalization.json'), 'w') as f:
            json.dump(reward_dict, f, indent=4)

    def read_reward_norm(self, reward_dir):
        with open(os.path.join(reward_dir, 'reward_normalization.json')) as f:
            reward_dict = json.load(f)
        self.gain = reward_dict['gain']
        self.bias = reward_dict['bias']

    def get_embedding(self, sentences):
        tokenized = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.max_input_len)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden_states = encoder_outputs["hidden_states"][-self.num_pooling_layers:]
        if self.num_pooling_layers > 1:
            last_hidden_states = torch.stack(last_hidden_states, dim=0).to(self.device) # shape: (n * bs * seq_length * hidden_dim)
        else:
            last_hidden_states = last_hidden_states[0].unsqueeze(0)

        # Mean pooling on last n hidden layers, following: https://arxiv.org/abs/2108.08877
        embedding = last_hidden_states.mean(dim=0).to(self.device) # shape: (bs * seq_length * hidden_dim)

        # Mask out hidden states generated from pad tokens
        attention_mask_unsqueezed = attention_mask.unsqueeze(-1) # shape: (bs * seq_length * 1)
        embedding = embedding * attention_mask_unsqueezed

        # Mean pooling
        embedding = embedding.sum(dim=1) # shape: (bs * hidden_dim)
        seq_lengths = attention_mask.sum(dim=-1)
        embedding = embedding / seq_lengths.unsqueeze(-1)
        
        # Apply L2 norm
        embedding = torch.nn.functional.normalize(embedding, p=2.0, dim=1)
        return embedding