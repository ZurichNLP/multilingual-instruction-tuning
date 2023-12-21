#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Gathers translated guanaco data into a single dataframe to construct a multilingual training set.


Example usage:
    python prepare_mt_guanaco_data.py

"""

from pathlib import Path
import pandas as pd

seed = 42
n = 200
data_dir = 'data/guanaco'

mono_en = pd.read_json(Path(data_dir) / 'guanaco_train_mono.json', lines=True)
mono_en.info()

# tgt_langs = ['en', 'ca', 'de', 'es', 'fr', 'ru', 'zh']
tgt_langs = [
    'en', # 3627
    'es', # 3850
    'ru', # 754
    'de', # 351
    'zh', # 330
    'fr', # 259
    'ca', # 250
]


mono_ids = set(mono_en['id'].tolist())
diff_ids = None
for lang in tgt_langs[1:]:
    ml6 = pd.read_json(Path(data_dir) f'guanaco_train_{lang}.json', lines=True)
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
to_translate_df = mono_en[mono_en['id'].isin(diffs)]
to_translate_df.info()

non_translated_df = mono_en[mono_en['id'].isin(common_ids)] # mono_en[~mono_en['id'].isin(diffs)]
non_translated_df.info()

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

translated_df['text'] = None
lang_i = 0
for i, row in translated_df.iterrows():
    
    if i > 0 and i % 200 == 0:
        lang_i += 1
    
    translated_df.at[i, 'text'] = row[non_eng_langs[lang_i]]
    translated_df.at[i, 'lang'] = non_eng_langs[lang_i]


ids = None
for i, lang in enumerate(non_eng_langs):

    # subsample eng_Latn only to match all langs for monolingual training
    non_en_data_to_add = translated_df[translated_df['lang'].isin(non_eng_langs[:i])][['text', 'lang', 'id']]

    en_data_to_add = translated_df[~translated_df['id'].isin(non_en_data_to_add['id'].tolist())][['en', 'id']].rename(columns={'en': 'text'})
    en_data_to_add['lang'] = 'en'

    print(f'adding {len(en_data_to_add)} rows of en data')
    # added_data = pd.concat([translated_df[lang].sample(n=200, random_state=seed) for i, lang in enumerate(non_eng_langs[:i])]).reset_index(drop=True)
    print(f'adding {len(non_en_data_to_add)} rows of non-en data')
    
    # concatenate mono_dataset and added_data
    ml_df = pd.concat([non_translated_df, en_data_to_add, non_en_data_to_add]).reset_index(drop=True)

    # ensure ids are unique
    assert len(ml_df['id'].unique()) == len(ml_df)
    
    # ensure ids are the same for each dataset version
    if not ids:
        ids = set(ml_df['id'].tolist())
    else:
        assert ids == set(ml_df['id'].tolist())

    print(ml_df['lang'].value_counts())

    outfile = Path(data_dir) / f'guanaco_train_mt_ml{i+1}.json'
    ml_df.to_json(outfile, orient='records', lines=True, force_ascii=False)
    print(f'Wrote {len(ml_df)} samples to {outfile}')

# 3. create ml LOO datasets
for i, lang in enumerate(non_eng_langs):
    
    other_langs = non_eng_langs[:i] + non_eng_langs[i+1:]
    print(lang, other_langs)

    c = 0
    # for each other lang, take n non-overlapping rows
    sdfs = []
    for other_lang in other_langs:
        sdf = translated_df.iloc[c:c+n][['id', other_lang]]
        sdf['lang'] = other_lang
        sdf.rename(columns={other_lang: 'text'}, inplace=True)
        print(len(sdf), sdf.columns)
        sdfs.append(sdf)
        c += n

    ml_df = pd.concat(sdfs + [non_translated_df], axis=0).reset_index(drop=True)
    
    # shuffle
    ml_df = ml_df.sample(frac=1, random_state=seed+1).reset_index(drop=True)

    # ensure ids are unique
    assert len(ml_df['id'].unique()) == len(ml_df)

    ml_df.to_json(Path(data_dir) / f'guanaco_train_mt_{lang}.json', orient='records', lines=True, force_ascii=False)
    print(f'wrote {len(ml_df)} rows to guanaco_train_mt_{lang}.json')
  
# # 3. create ml LOO datasets with overlapping ids - Not used
# for i, lang in enumerate(non_eng_langs):
    
#     other_langs = non_eng_langs[:i] + non_eng_langs[i+1:]
#     print(lang, other_langs)

#     c = 0
#     # for each other lang, take n non-overlapping rows
#     sdfs = []
#     for other_lang in other_langs:
#         sdf = translated_df.iloc[c:c+n][['id', other_lang]]
#         sdf['lang'] = other_lang
#         sdf.rename(columns={other_lang: 'text'}, inplace=True)
#         print(len(sdf), sdf.columns)
#         sdfs.append(sdf)
#         c += n
    
#     sdfs = pd.concat(sdfs, axis=0).reset_index(drop=True).sort_values(by=['id'])

#     # get rows from to_translate_df that match in sdfs    
#     x = to_translate_df[to_translate_df['id'].isin(sdfs['id'].tolist())].sort_values(by=['id'])

#     # get remainder of rows from non_translated_df
#     y = non_translated_df.sample(n=total_n - len(x) - len(sdfs), random_state=seed).reset_index(drop=True)

#     ml_df = pd.concat([sdfs, x, y], axis=0).reset_index(drop=True)
    
#     # shuffle
#     ml_df = ml_df.sample(frac=1, random_state=seed+1).reset_index(drop=True)

#     # ensure ids overlap as we want instances of the same id in different langs
#     assert len(ml_df['id'].unique()) != len(ml_df)

#     ml_df.to_json(Path(data_dir) / f'guanaco_train_mtol_{lang}.json', orient='records', lines=True, force_ascii=False)
#     print(f'wrote {len(ml_df)} rows to guanaco_train_mtol_{lang}.json')
