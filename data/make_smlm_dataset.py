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


def create_sentence_data(data):
    processed_data = []
    idx = 0

    for d in data:
        split_text = d['text'].split('.')#list(filter(None, re.split('.', d['text'])))
        if len(split_text) > 3:
            processed_data.append(add_punct(split_text))

    print(f"len of preprocessed data: {len(data)}")
    print(f"len of processed data: {len(processed_data)}")

    processed_data = format_to_mlm_data(processed_data)
    print(f"Done! Exapmle: \n{processed_data[0]}")
            

def main():
    random.seed(0)
    #load wikitext data
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    for split in dataset.keys():
        processed_data = create_sentence_data(dataset[split])
        fname = f'./smlm_{split}.json'
        with open(fname, 'w') as f:
            json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    main()