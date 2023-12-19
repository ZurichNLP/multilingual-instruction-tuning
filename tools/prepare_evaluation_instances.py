#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script to randomly sample evaluation instances from model outputs generated with multiple seeds.

Example usage:

    python prepare_evaluation_instances.py -i data/outputs/falcon_7b_de_merged/

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, help="path to folder containing model outputs files")
    ap.add_argument("-o", "--output_dir", required=False, default='data/to_annotate', help="path to folder to save evaluation instances")
    ap.add_argument("-n", "--num_instances", required=False, default=-1, help="number of evaluation instances to sample per model")
    ap.add_argument("--all_columns", required=False, default=False, action='store_true', help="include all columns in output file")
    ap.add_argument("-s", "--seed", required=False, default=42, help="random seed")
    return ap.parse_args()

def parse_filename(file):
    """
    data/outputs/llama_2_7b_hf_en_merged/alpaca_eval_instructions_ru-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.eval
    """
    # get test set and context, allowing for no context
    model_name = Path(file).parent.name
    parts = Path(file).stem.split('-')

    test_set = parts[0]
    context = parts[1]
    prompt = parts[2]
    seed = int(parts[3][1:])
    top_k = int(parts[4][1:])
    top_p = float(parts[5][1:])
    temp = float(parts[6][1:])
    bs = int(parts[7][1:])

    return model_name, test_set, prompt, seed

def get_generations(dir_path):
    dfs = []
    for generations_file in Path(dir_path).glob('*.jsonl'):
        # print(eval_file)
        df = pd.read_json(generations_file, lines=True)
        model_name, test_set, prompt, seed = parse_filename(generations_file)
        df['model_name'] = model_name
        df['test_set'] = test_set
        df['context'] = prompt
        df['seed'] = seed
        df['id'] = df.index
        dfs.append(df)
        
    df = pd.concat(dfs, axis=0, ignore_index=False)

    # move input_file, model_name and seed to front
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('model_name')))
    cols.insert(1, cols.pop(cols.index('test_set')))
    cols.insert(2, cols.pop(cols.index('seed')))
    # cols.insert(0, df.ndex)
    df = df.reindex(columns=cols)

    return df


if __name__ == '__main__':
    args = set_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = get_generations(args.input_dir)

    print(data['test_set'].value_counts())

    # for each test_set, sample N generations from each model
    for test_set, df in data.groupby('test_set'):
        
        df['seed'] = df['seed'].astype(str)
        # Pivot the dataframe to get 'seed' values as column headers
        pivot_df = df.pivot(index=['model_name', 'test_set', 'id', 'source', 'source_lang'], columns='seed', values=['system', 'system_lang']).reset_index()

        # merge the multilevel column headers
        pivot_df.columns = ['_'.join(col).strip('_') for col in pivot_df.columns.values]

        # get the unique seed values
        df['seed'] = df['seed'].astype(int)
        seeds = df['seed'].unique().tolist()

        # randomly select a seed for each row
        selected_seeds = np.random.choice(seeds, size=len(pivot_df))
        
        # add new columns for anno_seed and anno_response and anno_lang
        pivot_df['anno_seed'] = np.nan
        pivot_df['anno_system'] = ''
        pivot_df['anno_lang'] = ''

        for i, seed in enumerate(selected_seeds):
            pivot_df.at[i, 'anno_seed'] = int(seed)
            pivot_df.at[i, 'anno_system'] = pivot_df.at[i, (f'system_{seed}')]
            pivot_df.at[i, 'anno_lang'] = pivot_df.at[i, (f'system_lang_{seed}')]

        # convert anno_seed to int        
        pivot_df['anno_seed'] = pivot_df['anno_seed'].astype(int)

        assert len(df['model_name'].unique()) == 1
        model_name = df['model_name'].unique()[0]

        if args.all_columns:
            df = pivot_df
        else:
            df = pivot_df[['id', 'model_name', 'test_set', 'source', 'anno_system', 'source_lang', 'anno_lang', 'anno_seed']]

        if args.num_instances > 0:
            df = df.sample(n=min(args.num_instances, len(df)), random_state=args.seed)
        else:
            df = df.sample(frac=1, random_state=args.seed)

        output_file = Path(args.output_dir) / f'{model_name}-{test_set}-anno_instances.jsonl'
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        print(f'Saved {output_file}')

        