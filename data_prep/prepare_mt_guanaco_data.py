#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script creates multiple training sets with 1 to 6 languages.
The script will subsample n samples for each non-English language from the translated data subsets.
By default, n=200. For other values, pass the desired n as an argument to the script.

Example usage:
    python prepare_mt_guanaco_data.py

    # for other values of n, e.g. 10 use:
    python prepare_mt_guanaco_data.py 10

"""

import sys
from pathlib import Path
import pandas as pd

seed = 42
n = 200
data_dir = 'data/guanaco'

tgt_langs = [
    'en', # 3627
    'es', # 3850
    'ru', # 754
    'de', # 351
    'zh', # 330
    'fr', # 259
    'ca', # 250
]

if len(sys.argv) > 1:
    n = int(sys.argv[1])
print(f'Using {n} samples per language')

# 1. load the full EN monolingual guanaco data
mono_en = pd.read_json(Path(data_dir) / 'guanaco_train_mono.json', lines=True)
mono_en.info()
mono_ids = set(mono_en['id'].tolist())

def replace_role_labels(text):
    text = text.replace('<S1>', '### Human:')
    text = text.replace('<S2>', '### Assistant:')
    return text

# load translated guanaco data:
# guanaco_train_mono_1k_ca.json, guanaco_train_mono_1k_de.json, guanaco_train_mono_1k_es.json, etc.
dfs = []
for lang in tgt_langs:
    df = pd.read_json(Path(data_dir) / f'guanaco_train_mono_1k_{lang}.json', lines=True)
    df['text'] = df['text'].apply(replace_role_labels)
    df.rename(columns={'text': f'{lang}'}, inplace=True)
    dfs.append(df)
    
# merge all dataframes horizontally
translated_df = pd.concat(dfs, axis=1)
translated_df.info()

# 2. create "multilingual" datasets with 1, 2, 3, 4+ langs
total_n = len(mono_en)
non_eng_langs = [lang for lang in tgt_langs if lang != 'en']

# shuffle data
translated_df = translated_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# if n is less than 200, subsample the data to the total number of non-eng samples
if n < 200:
    translated_df = translated_df[:n * len(non_eng_langs)]

translated_df['text'] = None
lang_i = 0

for i, row in translated_df.iterrows():
    
    if i > 0 and i % n == 0:
        lang_i += 1
    
    translated_df.at[i, 'text'] = row[non_eng_langs[lang_i]]
    translated_df.at[i, 'lang'] = non_eng_langs[lang_i]

translated_ids = set(translated_df['id'].tolist())

# remove translated samples from mono_en
non_translated_df = mono_en[~mono_en['id'].isin(translated_ids)].reset_index(drop=True)

ids = None
for i, lang in enumerate(non_eng_langs):

    # subsample eng_Latn only to match all langs for monolingual training
    non_en_data_to_add = translated_df[translated_df['lang'].isin(non_eng_langs[:i])][['text', 'lang', 'id']]

    en_data_to_add = translated_df[~translated_df['id'].isin(non_en_data_to_add['id'].tolist())][['en', 'id']].rename(columns={'en': 'text'})
    print(len(en_data_to_add))

    en_data_to_add['lang'] = 'en'

    print(f'adding {len(en_data_to_add)} rows of en data')
    # added_data = pd.concat([translated_df[lang].sample(n=200, random_state=seed) for i, lang in enumerate(non_eng_langs[:i])]).reset_index(drop=True)
    print(f'adding {len(non_en_data_to_add)} rows of non-en data')
    
    # concatenate mono_dataset and added_data
    ml_df = pd.concat([non_translated_df, en_data_to_add, non_en_data_to_add]).reset_index(drop=True)

    print(ml_df['lang'].value_counts())

    # breakpoint()
    # ensure ids are unique
    assert len(ml_df['id'].unique()) == len(ml_df)
    
    # ensure ids are the same for each dataset version
    if not ids:
        ids = set(ml_df['id'].tolist())
    else:
        assert ids == set(ml_df['id'].tolist())

    print(ml_df['lang'].value_counts())

    if n == 200:
        outfile = Path(data_dir) / f'guanaco_train_mt_ml{i+1}.json'
    # ablation study train sets - only for ml3
    elif n < 200 and i == 2:
        outfile = Path(data_dir) / f'guanaco_train_mt_ml{i+1}_n{n}.json'
    else:
        outfile = None
    
    if outfile:
        ml_df.to_json(outfile, orient='records', lines=True, force_ascii=False)
        print(f'Wrote {len(ml_df)} samples to {outfile}')
