#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from https://github.com/bjoernpl/lm-evaluation-harness-de/blob/mmlu_de/eval_de.py
"""

import os
import argparse
import json
import logging
from pathlib import Path
import random
from time import time

from lm_eval import tasks, evaluator, utils
import lm_eval.models
import numpy as np
import pandas as pd

logging.getLogger("openai").setLevel(logging.WARNING)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default="data/lm_evals")
    parser.add_argument("--skip_fewshots", type=list, default=[])
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

tasks_per_fewshot = {
    # 5: [
    #     "hendrycksTest*",
    #     "MMLU-DE*",
    # ],
    # 10: [
    #     "hellaswag",
    #     "hellaswag_de"
    # ],
    # 25: [
    #     "arc_challenge",
    #     "arc_challenge_de"
    # ],
    0: [
        "xwinograd_en", # ["en", "fr", "jp", "pt", "ru", "zh"]
        "xwinograd_fr",
        "xwinograd_jp",
        "xwinograd_pt",
        "xwinograd_ru",
        "xwinograd_zh",
        "pawsx_en", # ["en", "de", "es", "fr", "ja", "ko", "zh",]
        "pawsx_de",
        "pawsx_es",
        "pawsx_fr",
        "pawsx_ja",
        "pawsx_ko",
        "pawsx_zh",
        "xnli_en", # ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh",]
        "xnli_de",
        "xnli_es",
        "xnli_fr",
        "xnli_ru",
        "xnli_zh",
        "xnli_bg",
        "xnli_el",
        "xnli_hi",
    ]
}


def main():
    args = parse_args()
    
    # create output path
    Path(args.output_base_path).mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "config": {"model": args.model, "model_args": args.model_args},
        "results": {},
        "versions": {},
    }

    model_name = Path(args.model_args.split(',')[0].split('=')[1]).name


    for num_fewshots, task_list in tasks_per_fewshot.items():
        
        start = time()

        task_names = utils.pattern_match(task_list, tasks.ALL_TASKS)

        if args.debug:
            random.seed(0)
            task_names = random.sample(task_names, 4)

        print(
            f"Running:\n"
            f"{args.model} ({args.model_args})\n"
            f"limit: {args.limit}\n"
            f"provide_description: {args.provide_description}\n"
            f"num_fewshot: {args.num_fewshot}\n"
            f"batch_size: {args.batch_size}\n"
            f"device: {args.device}\n"
            f"no_cache: {args.no_cache}\n"
            f"tasks: {task_names}\n"
        )

        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=num_fewshots,
            batch_size=args.batch_size,
            device=args.device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=None,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
            bootstrap_iters=100000
        )

        all_results["results"].update(results["results"])

        all_results["versions"].update(results["versions"])

        all_results["config"] = results["config"]
        
        time_taken = time() - start
        all_results["time_taken"] = time_taken

        dumped = json.dumps(all_results, indent=2)
        
        print(dumped)

        output_file = Path(args.output_base_path) / f'{model_name}_fs{num_fewshots}.json'
        with open(output_file, "w", encoding='utf8') as f:
            f.write(dumped)
        print(f'Wrote results {num_fewshots}-shot results to {output_file} in {time_taken} seconds.')

    dumped = json.dumps(all_results, indent=2)
    
    print(dumped)

    output_file = Path(args.output_base_path) / f'{model_name}.json'
    with open(output_file, "w", encoding='utf8') as f:
        f.write(dumped)
    print(f'Wrote results to {output_file}')

    


if __name__ == "__main__":
    main()