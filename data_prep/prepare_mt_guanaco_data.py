#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script creates multiple training sets with 1 to 6 languages.
The script will subsample n samples for each non-English language from the translated data subsets.
By default, n=200. For other values, pass the desired n as an argument to the script.

Example usage:
    python data_prep/prepare_mt_guanaco_data.py --strategy incremental_1 --n 200

    # for other values of n, e.g. 10 use:
    python data_prep/prepare_mt_guanaco_data.py --strategy incremental_1 --n 10

    # for ablation study 1, use:
    python data_prep/prepare_mt_guanaco_data.py --strategy abl_1

    # for ablation study 2, use:
    python data_prep/prepare_mt_guanaco_data.py --strategy abl_2 --n 300
    

"""

import sys
from pathlib import Path
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/guanaco', help='path to guanaco data dir')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--n', type=int, default=None, help='number of samples per language')
parser.add_argument('--strategy', type=str, default='incremental_1', help='data preparation strategy')
args = parser.parse_args()

n = args.n
strategy = args.strategy
seed = args.seed
data_dir = args.data_dir

tgt_langs = [
    'en', # 3627
    'es', # 3850
    'ru', # 754
    'de', # 351
    'zh', # 330
    'fr', # 259
    # 'ca', # 250
]

def replace_role_labels(text):
    text = text.replace('<S1>', '### Human:')
    text = text.replace('<S2>', '### Assistant:')
    return text

# 1. load the full EN monolingual guanaco data
mono_en = pd.read_json(Path(data_dir) / 'guanaco_train_mono.json', lines=True)
mono_en.info()
mono_ids = set(mono_en['id'].tolist())

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
translated_ids = set(translated_df['id'].tolist())

if strategy == 'incremental_1' and n:

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

# abl_1: increment langauges keeping a fixed number of non-eng samples
elif strategy == 'abl_1':

    # remove translated samples from mono_en
    non_translated_df = mono_en[~mono_en['id'].isin(translated_ids)].reset_index(drop=True)
    
    # total number of samples for each language
    # total_non_eng = 1000
    total_non_eng = 400

    for k in range(len(non_eng_langs)):
        n = total_non_eng // (k + 1)    
        print(f'Using {n} samples for {non_eng_langs[:k+1]}')        

        translated_df_k = translated_df[non_eng_langs[:k+1] + ['id']][:total_non_eng]
        print(translated_df_k.columns)

        # get the remaining samples from english if needed
        en_data_to_add = translated_df[~translated_df['id'].isin(translated_df_k['id'].tolist())][['en', 'id']].rename(columns={'en': 'text'})
        en_data_to_add['lang'] = 'en'

        translated_df_k['text'] = None
        translated_df_k['lang'] = None
        ids = None
        lang_i = 0
        for i, row in translated_df_k.iterrows():
            
            # increment lang_i every n samples to switch to the next language
            if i > 0 and i % n == 0 and i != total_non_eng - 1:
                # skip the last one incase it's not a multiple of n (e.g. 333)
                lang_i += 1
            
            translated_df_k.at[i, 'text'] = row[non_eng_langs[lang_i]]
            translated_df_k.at[i, 'lang'] = non_eng_langs[lang_i]

        # concatenate mono_dataset and added_data        
        ml_df = pd.concat([non_translated_df, translated_df_k[['text', 'lang', 'id']], en_data_to_add]).reset_index(drop=True)

        # ensure ids are unique
        assert len(ml_df['id'].unique()) == len(ml_df)
        
        # ensure ids are the same for each dataset version
        if not ids:
            ids = set(ml_df['id'].tolist())
        else:
            assert ids == set(ml_df['id'].tolist())

        print(ml_df['lang'].value_counts())
        outfile = Path(data_dir) / f'guanaco_train_mt_ml{k+2}_n{n}.json'
        ml_df.to_json(outfile, orient='records', lines=True, force_ascii=False)
        print(f'Wrote {len(ml_df)} samples to {outfile}')

# abl_2: increment non-eng sample keeping a fixed number of languages
elif strategy == 'abl_2' and n:

    # subsample the translated data to n samples
    translated_df_l = translated_df[:n]
    lang = non_eng_langs[0]
    translated_df_l['text'] = translated_df_l[lang]
    translated_df_l['lang'] = [lang]*len(translated_df_l)

    # remove translated samples from mono_en
    non_translated_df = mono_en[~mono_en['id'].isin(translated_ids)].reset_index(drop=True)
    
    # subsample eng_Latn only to match all langs for monolingual training
    non_en_data_to_add = translated_df_l[['text', 'lang', 'id']]

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
    
    ids = None

    # ensure ids are the same for each dataset version
    if not ids:
        ids = set(ml_df['id'].tolist())
    else:
        assert ids == set(ml_df['id'].tolist())

    print(ml_df['lang'].value_counts())

    # ablation study train sets - only for ml2
    outfile = Path(data_dir) / f'guanaco_train_mt_ml2_n{n}.json'
    
    if Path(outfile).exists():
        raise ValueError(f'{outfile} already exists!')
    else:
        ml_df.to_json(outfile, orient='records', lines=True, force_ascii=False)
        print(f'Wrote {len(ml_df)} samples to {outfile}')

else:
    raise ValueError('Please pass a valid strategy and n')