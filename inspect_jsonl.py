#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Simple script to randomly inspect model outputs. 
Expects a JSONL input file with items containing input prompts and model outputs.

Example usage:
    
    python inspect_outputs \
        data/outputs/llama_2_7b_hf_mono_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s42-k50-p0.9-t0.8-b8.jsonl \
        --shuffled \
        --seed 0 \
        --ignore_keys finish_reason secs contexts

"""

from pathlib import Path
import pandas as pd
import argparse
import random
from typing import List, Dict, Optional

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', default=None, type=str, help='Path to the file containing model generated outputs')
    parser.add_argument('--ignore_keys', nargs='+', default=['finish_reason', 'secs', 'contexts'], help='List of keys to ignore when inspecting model outputs')
    parser.add_argument('--shuffled', action='store_true', help='Randomly sample from the input file')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser.parse_args()

def pretty_print_instance(example: Dict, idx, ignore_keys: Optional[List] =None) -> None:
    
    for key in example.keys():
        if ignore_keys and key in ignore_keys:
            continue
        print(f"\n**** {idx} {key.upper()} ****\n")
        print(f"{example[key]}")

    print("="*80)

def peek_outputs(data, shuffled=False, seed=42, ignore_keys=None):
    
    if shuffled:
        random.seed(seed)
        data = data.sample(frac=1, random_state=seed)
    
    for i, row in data.iterrows():
        pretty_print_instance(row.to_dict(), i, ignore_keys=ignore_keys)
        cont = input('Press enter to continue, or q to quit: ')
        if cont == 'q':
            break

if __name__ == '__main__':
    args = get_args()

    df = pd.read_json(args.infile, lines=True)

    print(f"Loaded {len(df)} examples from {args.infile}")

    peek_outputs(df, args.shuffled, args.seed, args.ignore_keys)
