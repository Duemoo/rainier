import json
import re
import os
import tqdm
import random
import numpy
from datasets import load_dataset


def add_punct(split_text):
    res = []
    for text in split_text[:-1]:
        res.append(text + '.')
    return res


def format_to_mlm_data(data):
    processed_data = []
    for i, d in enumerate(data):
        mask_idx = random.choice(range(1,len(d)-1))
        target = '<extra_id_0>' + d[mask_idx] + '<extra_id_1></s>'
        d[mask_idx] = '<extra_id_0>'
        masked_input = ''.join(d) + '<extra_id_1>'
        data_dict = {'id': i, 'input': masked_input, 'target': target}
        processed_data.append(data_dict)

    return processed_data


def format_to_lm_data(data):
    processed_data = []
    for i, d in enumerate(data):
        question = ''.join(d[:-2])
        target = d[-1]
        data_dict = {'id': i, 'input': question, 'target': target}
        processed_data.append(data_dict)

    return processed_data


def create_sentence_data(data, mode):
    processed_data = []
    idx = 0

    for d in data:
        split_text = d['text'].split('.')#list(filter(None, re.split('.', d['text'])))
        if len(split_text) > 3:
            processed_data.append(add_punct(split_text))

    print(f"len of preprocessed data: {len(data)}")
    print(f"len of processed data: {len(processed_data)}")

    if mode=='smlm':
        processed_data = format_to_mlm_data(processed_data)
    elif mode=='slm':
        processed_data = format_to_lm_data(processed_data)
    else:
        raise NotImplementedError

    print(f"Done! Exapmle: \n{processed_data[0]}")

    return processed_data
            

def main(mode):
    random.seed(0)
    #load wikitext data
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    for split in dataset.keys():
        processed_data = create_sentence_data(dataset[split], mode)
        fname = f'./{mode}/{mode}_{split}.json'
        with open(fname, 'w') as f:
            json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    mode = 'slm'
    main(mode)