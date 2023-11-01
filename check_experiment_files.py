#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python check_experiment_files.py -i data/outputs

"""

import argparse
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

def get_expected_test_sets(dir_path):
    expected_test_sets = [f.stem for f in Path(dir_path).glob('alpaca_eval_instructions_*.json')]
    return expected_test_sets

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

def main(dir_path: str):
    expected_seeds = [0, 42, 723]
    expected_test_sets = get_expected_test_sets(Path(dir_path).parent)
    print(f"*** {len(expected_test_sets) * len(expected_seeds)} expected total inference runs for test sets {expected_test_sets} and seeds {expected_seeds} ***")

    model_files = {}
    for model_outputs_dir in Path(dir_path).iterdir():
        model_files[model_outputs_dir.name] = gather_existing_files(model_outputs_dir)

        # # check that all expected test sets * expected seeds are present
        # for test_set in expected_test_sets:
        #     for seed in expected_seeds:
        #         for filename in model_files[model_outputs_dir.name]['output_files']:
        #             _, ftest_set, _, fseed = parse_filename(filename)
        #             breakpoint()
        #             if test_set == ftest_set and seed == fseed:
        #                 break
        #         print(f"Missing eval file for {test_set} and seed {seed} in {model_outputs_dir.name}")
            
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--indir', default='data/outputs', type=str, help='Directory containing experiment results')
    args = ap.parse_args()


    main(args.indir)
