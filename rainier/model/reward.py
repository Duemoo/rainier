import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
from itertools import chain
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from utils.utils import reduce_mean


class Reward:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 max_input_len,
                 batch_size,
                 reward_shape,
                 kl_coef,
                 ensembling,
                 device: torch.device,
                ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_type)
        self.inference_model = SentenceTransformer(model_ckpt if model_ckpt is not None else model_type)
        #self.inference_model = T5ForConditionalGeneration.from_pretrained(model_ckpt if model_ckpt is not None else model_type)
        self.inference_model.eval()
        self.inference_model.to(device)

        self.gain, self.bias = None, None
        self.max_input_len = max_input_len
        self.batch_size = batch_size
        self.reward_shape = reward_shape
        self.kl_coef = kl_coef
        self.ensembling = ensembling

        self.device = device

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='none')

    """
    questions: list of strings
    knowledges: list of knowledges, 1 to 1 mapping to questions
    choicess: list of lists of candidate choices for each question
    answer_ixs: list of integer indeces corresponding to ground truth answer index from answers list of lists
    """
    def get_reward(self,
                   questions: List[str],
                   knowledges: List[str],
                   answer: List[int],
                   override_gain = None,
                   override_bias = None,
                   skip_reward = False,
                  ) -> Tuple[List[float], float, int, int]:
        if knowledges is None:
            knowledges = [None for _ in questions]

        assert len(questions) == len(knowledges)

        questions = [a.lower() for a in questions]
        knowledges = [a.lower() if a is not None else None for a in knowledges]
        answers = [[a.lower() for a in b] for b in answer]
        
        rewards_raw = []
        for k, a in zip(knowledges, answers):
            k_emb = self.inference_model.embed(k)
            a_emb = self.inference_model.embed(a)
            rewards_raw.append(util.dot_score(k_emb, a_emb))

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

    # knlowledge별로 inference를 했을 때, 최종 결과물을 어떻게 내보낼 것이냐에 대한 함수 => reward가 가장 높은 결과를 만들게 하거나, 아니면 애초에 inference에서는 하나만 만들도록 변경
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

