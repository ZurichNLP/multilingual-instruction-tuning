#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python check_experiment_files.py -i data/outputs

"""

import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

def quick_lc(file: Path):
    with open(file, 'r') as f:
        lc = sum([1 for _ in f])
    return lc

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

def gather_existing_files(model_outputs_dir: str):
    files = {'eval_files': [], 'llm_eval_files': [], 'param_files': [], 'output_files': []}
    
    for file in model_outputs_dir.iterdir():
        if file.suffix == '.eval':
            files['eval_files'].append(file.name)
        elif file.suffix == '.json':
            files['param_files'].append(file.name)
        elif file.suffix == '.jsonl':
            files['output_files'].append(file.name)
        elif file.suffix == '.llm_eval':
            files['llm_eval_files'].append(file.name)
        
    print(f"{model_outputs_dir}: Outputs: {len(files['output_files'])}\tEvals: {len(files['eval_files'])}\tLLM Evals: {len(files['llm_eval_files'])}\tParams: {len(files['param_files'])}")
    return files

def main(dir: str):
    model_files = {}
    for model_outputs_dir in Path(dir).iterdir():
        model_files[model_outputs_dir.name] = gather_existing_files(model_outputs_dir)
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--indir', default='data/outputs', type=str, help='Directory containing experiment results')
    args = ap.parse_args()

    main(args.indir)
