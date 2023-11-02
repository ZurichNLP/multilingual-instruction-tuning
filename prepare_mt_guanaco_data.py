#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Gathers translated guanaco data into a single dataframe to construct a multilingual training set.


Example usage:
    python prepare_mt_guanaco_data.py

"""

import pandas as pd

seed = 42
n = 200

mono_en = pd.read_json('data/guanaco_train_mono.json', lines=True)
mono_en.info()

tgt_langs = ['en', 'ca', 'de', 'es', 'fr', 'ru', 'zh']

mono_ids = set(mono_en['id'].tolist())
diff_ids = None
for lang in tgt_langs[1:]:
    ml6 = pd.read_json(f'data/guanaco_train_{lang}.json', lines=True)
    ml6 = ml6[ml6['lang'] == 'en']
    common_ids = mono_ids.intersection(ml6['id'].tolist())
    # get ids not in common
    diffs = mono_ids.difference(set(ml6['id'].tolist()))
    if diff_ids is None:
        diff_ids = diffs
    elif diffs != diff_ids:
        print('WARNING: diffs not the same!')
    
    print(lang, len(common_ids), len(diff_ids))

# get the rows from mono_en that are not in common_ids for translation into other langs
to_translate = mono_en[mono_en['id'].isin(diffs)]
to_translate.info()

def replace_role_labels(text):
    text = text.replace('<S1>', '### Human:')
    text = text.replace('<S2>', '### Assistant:')
    return text

# load translated guanaco data:
# guanaco_train_mono_1k_ca.json, guanaco_train_mono_1k_de.json, guanaco_train_mono_1k_es.json, etc.
dfs = []
for lang in tgt_langs:
    df = pd.read_json(f'data/guanaco_train_mono_1k_{lang}.json', lines=True)
    df['text'] = df['text'].apply(replace_role_labels)
    df.rename(columns={'text': f'{lang}'}, inplace=True)
    dfs.append(df)
    
# merge all dataframes horizontally
df = pd.concat(dfs, axis=1)
df.info()

en_sdf = mono_en[mono_en['id'].isin(common_ids)]
en_sdf.info()

for i, lang in enumerate(tgt_langs[1:]):
    
    other_langs = tgt_langs[1:][:i] + tgt_langs[1:][i+1:]
    print(lang, other_langs)

    c = 0
    # for each other lang, take n non-overlapping rows
    sdfs = []
    for other_lang in other_langs:
        sdf = df.iloc[c:c+n][['id', other_lang]]
        sdf['lang'] = other_lang
        sdf.rename(columns={other_lang: 'text'}, inplace=True)
        print(len(sdf), sdf.columns)
        sdfs.append(sdf)
        c += n

    ml_df = pd.concat(sdfs + [en_sdf], axis=0).reset_index(drop=True)
    
    # shuffle
    ml_df = ml_df.sample(frac=1, random_state=seed+1).reset_index(drop=True)

    # ensure ids are unique
    assert len(ml_df['id'].unique()) == len(ml_df)

    ml_df.to_json(f'data/guanaco_train_mt_{lang}.json', orient='records', lines=True, force_ascii=False)

    print(f'wrote {len(ml_df)} rows to data/guanaco_train_mt_{lang}.json')
    