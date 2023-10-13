#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example usage:

    python prepare_guanaco.py --dataset_path data/guanaco_train.json

"""

import argparse
import re
from pathlib import Path
import json
import pandas as pd
from datasets import load_dataset

# mapping from open-lid lang codes to IETF lang codes
# https://github.com/laurieburchell/open-lid-dataset/blob/main/languages.md
lang_set = {
    'spa_Latn': 'es', # 3850
    'eng_Latn': 'en', # 3627
    'rus_Cyrl': 'ru', # 754
    'deu_Latn': 'de', # 351
    'zho_Hans': 'zh', # 330
    'fra_Latn': 'fr', # 259
    'cat_Latn': 'ca', # 250
    'tha_Thai': 'th', # 167
    'por_Latn': 'pt', # 164
    'ita_Latn': 'it', # 113
}

def set_args():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--dataset_path", required=False, default="data/guanaco_train.json", help="path to dataset with lang column")
    ap.add_argument("--tgt_lang", required=False, default=None, help="target language")
    ap.add_argument("--seed", required=False, default=42, help="random seed")

    return ap.parse_args()

def main(args):

    dataset = pd.read_json(args.dataset_path, lines=True)

    # subsample dataset eng: 2000, others 200
    lang_datasets = {}
    for lang in lang_set:
        if lang == 'eng_Latn':
            lang_datasets[lang] = dataset[dataset['lang'] == lang].sample(n=2200, random_state=args.seed)
        elif len(dataset[dataset['lang'] == lang]) > 200:
            lang_datasets[lang] = dataset[dataset['lang'] == lang].sample(n=200, random_state=args.seed)
        else:
            pass

    # subsample eng_Latn only to match all langs for monolingual training
    total_n = sum([len(x) for x in lang_datasets.values()])
    mono_dataset = dataset[dataset['lang'] == 'eng_Latn'].sample(n=total_n, random_state=args.seed)
    mono_dataset['lang'] = mono_dataset['lang'].replace(lang_set)
    outfile = Path(args.dataset_path).parent / f'{Path(args.dataset_path).stem}_mono.json'
    mono_dataset.to_json(outfile, orient='records', lines=True, force_ascii=False)

    # concatenate all lang_datasets    
    dataset = pd.concat(lang_datasets.values()).reset_index(drop=True)

    # rename open-lid lang codes to IETF lang codes
    dataset['lang'] = dataset['lang'].replace(lang_set)

    print(dataset['lang'].value_counts())

    for lang in dataset['lang'].unique():
        print(f"Removing {len(dataset[dataset['lang'] == lang])} samples of {lang} ...")
        if lang == 'en':
            # randomly remove 200 eng_Latn samples
            sub_dataset = dataset.drop(dataset[dataset['lang'] == lang].sample(n=200, random_state=args.seed).index)
        else:
            # drop all samples of current lang
            sub_dataset = dataset.drop(dataset[dataset['lang'] == lang].index)

        print(sub_dataset['lang'].value_counts())

        outfile = Path(args.dataset_path).parent / f'{Path(args.dataset_path).stem}_{lang[:2]}.json'
        sub_dataset.to_json(outfile, orient='records', lines=True, force_ascii=False)
        print(f'Wrote {len(sub_dataset)} samples to {outfile}')
        print()    
    return

if __name__ == "__main__":
    args = set_args()
    print(args)
    main(args)