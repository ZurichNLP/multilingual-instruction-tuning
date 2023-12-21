#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:
    python assign_guanaco_langs.py

"""

import argparse
import pandas as pd
from datasets import load_dataset
from open_lid import LIDModel

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-d", "--dataset", required=False, default="timdettmers/openassistant-guanaco", help="path to input dataset")
    ap.add_argument("-f", "--fasttext_model_path", required=False, default="resources/lid/lid201-model.bin", help="path to fasttext model")

    return ap.parse_args()

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

    lid = LIDModel()
    
    # assign lang column
    dataset['lang'], dataset['confidence'] = zip(*dataset.apply(lambda row: lid.predict(row['text']), axis=1))
    
    dataset['id'] = dataset.index

    # save dataset with lang column
    for split in ['train', 'test']:
        dataset[dataset['split'] == split][['text', 'lang', 'confidence', 'id']].to_json(f'./data/guanaco_{split}.json', orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    args = set_args()
    main(args)