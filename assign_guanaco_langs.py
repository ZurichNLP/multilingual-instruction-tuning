#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:
    python assign_guanaco_langs.py

"""

import argparse
import re

import json
import pandas as pd
from datasets import load_dataset
from collections import Counter
import numpy as np
import fasttext

lang_set = [
    'spa_Latn', # 3850
    'eng_Latn', # 3627
    'rus_Cyrl', # 754
    'deu_Latn', # 351
    'zho_Hans', # 330
    'fra_Latn', # 259
    'cat_Latn', # 250
    'tha_Thai', # 167
    'por_Latn', # 164
    'ita_Latn', # 113
]

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-d", "--dataset", required=False, default="timdettmers/openassistant-guanaco", help="path to input dataset")
    ap.add_argument("-f", "--fasttext_model_path", required=False, default="../llm_dqa/llm_dqa/resources/eval_models/openLID/lid201-model.bin", help="path to fasttext model")

    return ap.parse_args()

def assign_lang_id(texts, model):
    langs = []
    for text in texts:
        text = re.sub('\n', ' ', text)
        text = re.sub('(### Human:| ### Assistant:)', ' ', text).strip()
        lang = model.predict(text)[0][0].split("__")[-1]
        langs.append(lang)
    return langs

def main(args):

    dataset = load_dataset(args.dataset)

    # load dataset as pandas dataframe
    dataset_train = pd.DataFrame(dataset['train'])
    dataset_test = pd.DataFrame(dataset['test'])

    # add split column
    dataset_train['split'] = 'train'
    dataset_test['split'] = 'test'

    # concatenate train and test
    dataset = pd.concat([dataset_train, dataset_test]).reset_index(drop=True)

    # assign lang column
    dataset['lang'] = assign_lang_id(dataset['text'].to_list(), fasttext.load_model(args.fasttext_model_path))
    dataset['id'] = dataset.index

    # save dataset with lang column
    for split in ['train', 'test']:
        dataset[dataset['split'] == split][['text', 'lang', 'id']].to_json(f'./data/guanaco_{split}.json', orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    args = set_args()
    main(args)