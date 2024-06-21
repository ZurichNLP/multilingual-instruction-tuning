
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script is used to evaluate a set of responses using LLM as a judge for a 6-point Likert scale assessment.

Example usage:
    
    # direct eval in original language
    CUDA_VISIBLE_DEVICES=0 python llm_judge_prometheus.py \
        --input_file "resources/outputs/alpaca_eval/llama_2_7b_hf_ml6_merged/alpaca_eval_instructions_en-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl" \
        --eval_model_name "prometheus-eval/prometheus-7b-v2.0" \
        --src_key "source" \
        --tgt_key "system" \
        --data_seed 42

"""

import sys
import argparse
from pathlib import Path
import time
import pandas as pd
import json
from tqdm import tqdm

from helpers import logger
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE



# adapted from LIMA evaluation prompt for helpfulness
rubric_data = {
  "criteria": "You are evaluating a response that has been submitted for a particular task, using a specific set of standards. How helpful is the response in addressing the user's question and providing useful information?",
  "score1_description": "Not helpful - The generated text is completely irrelevant, unclear, or incomplete. It does not provide any useful information to the user.",
  "score2_description": "Somewhat helpful - The generated text has some relevance to the user's question, but it may be unclear or incomplete. It provides only partial information, logical inconsistencies, or the information provided may not be useful for the user's needs.",
  "score3_description": "Moderately helpful - The generated text is relevant to the user's question, and it provides a clear and complete answer. However, it may lack detail or explanation that would be helpful for the user.",
  "score4_description": "Helpful - The generated text is quite relevant to the user's question, and it provides a clear, complete, and detailed answer. It offers additional information or explanations that are useful for the user. However, some of the points of the response are somewhat repetitive or could be combined for greater clarity and concision.",
  "score5_description": "Very helpful - The generated text is highly relevant to the user's question, and it provides a clear, complete, and detailed answer. It offers additional information, explanations, or analogies that are not only useful but also insightful and valuable to the user. However, the structured of the response is not well-organized and there is no clear progression or logical sequence of different points in the response.",
  "score6_description": "Highly helpful - The generated text provides a clear, complete, and detailed answer. It offers additional information or explanations that are not only useful but also insightful and valuable to the user. The response is also in a logical and easy-to-follow manner by explicitly using headings, bullet points, or numbered lists to break up the information and make it easier to read.",
}


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, type=str, help="Path to input file")
    ap.add_argument("--output_file", required=False, type=str, help="Path to output file")
    ap.add_argument("--output_dir_base", required=False, default='data/llm_judge', type=str, help="Path to output directory")
    ap.add_argument("--eval_model_name", required=False, type=str, default="gpt-3.5-turbo-1106", help="Name of the model to use")
    ap.add_argument("--eval_type", required=False, type=str, default="likert", help="Evaluation type (abs|likert)")
    ap.add_argument("--verbose", action="store_true", default=False, help="Verbose")
    ap.add_argument("--limit", required=False, type=int, default=-1, help="Limit number of items to evaluate")
    ap.add_argument("--src_key", required=False, type=str, default="source", help="Source key")
    ap.add_argument("--tgt_key", required=False, type=str, default="system", help="Target key")
    ap.add_argument("--force", action="store_true", default=False, help="Overwrites existing outputs if found, otherwise skip.")
    ap.add_argument("--data_seed", required=False, type=int, default=42, help="Random seed for sampling items from data")
    ap.add_argument("--api_seed", required=False, type=int, default=42, help="Random seed for generation with API")
    ap.add_argument("--temperature", required=False, type=float, default=0.0, help="Temperature for generation with API")
    # ap.add_argument("--max_tokens", required=False, type=int, default=1000, help="Max tokens to use")
    ap.add_argument("--timeout", required=False, type=int, default=90, help="Timeout for API calls")
    ap.add_argument("--max_retries", required=False, type=int, default=10, help="Max retries for API calls for each item")
    ap.add_argument("--max_parallel_calls", required=False, type=int, default=10, help="Max parallel calls to API")
    return ap.parse_args()    

    

if __name__ == "__main__":

    # start timer
    start_time = time.time()

    args = set_args()

    data = pd.read_json(args.input_file, lines=True)

    # add id column if not present    
    if 'id' not in data.columns:
        data['id'] = data.index
    
    # limit number of items to evaluate
    if args.limit > 0:
        data = data.sample(args.limit, random_state=args.data_seed).reset_index(drop=True)

    logger.info(f"Evaluating {len(data)} items from {args.input_file}")

    # convert dataframe to dict
    data = data.to_dict(orient='records')

    # set output file
    if args.output_file:
        output_file = args.output_file
    
    elif args.output_dir_base:
        # e.g. data/llm_evals/likert/gpt-3.5-turbo-1106/llama_2_7b_hf_de_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8-translated.jsonl
        output_dir = Path(args.output_dir_base) / f"{args.eval_type}" / Path(args.eval_model_name).name / Path(args.input_file).parent.name
        output_file = Path(output_dir) / f"{Path(args.input_file).stem}-l{len(data)}-ds{args.data_seed}-as{args.api_seed}-{args.src_key}-{args.tgt_key}.jsonl"
    
    else:
        raise NotImplementedError("Please specify either --output_file or --output_dir_base.")
    
    logger.info(f"Output file: {output_file}")
    
    if Path(output_file).exists() and not args.force:
        logger.error(f"Output file already exists. Use --force to overwrite.")
        sys.exit(0)

    if args.eval_type != "likert":
        raise NotImplementedError

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if args.eval_model_name == "prometheus-eval/prometheus-8x7b-v2.0":
        logger.info("Using 2 GPUs for Prometheus 8x7b model ...")
        judge = PrometheusEval(model_id=args.eval_model_name, absolute_grade_template=ABSOLUTE_PROMPT, num_gpus=2)
    else:
        judge = PrometheusEval(model_id=args.eval_model_name, absolute_grade_template=ABSOLUTE_PROMPT, num_gpus=1)

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)    
    feedbacks, scores = judge.absolute_grade(
        instructions=[item[args.src_key] for item in data],
        responses=[item[args.tgt_key] for item in data],
        rubric=score_rubric,
        params={},
        reference_answers=None,
    )

    assert len(feedbacks) == len(scores) == len(data), f"Length mismatch: {len(feedbacks)} != {len(scores)} != {len(data)}"

    c = 0
    with open(output_file, "w+", encoding="utf8") as outf:
        for i, item in enumerate(data):
            item["eval_reasoning"] = feedbacks[i]
            item["eval_score"] = scores[i]
            item["eval_meta"] = {"model_name": args.eval_model_name}
            outf.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            c += 1


    logger.info(f"Done. Wrote {c} items to {output_file} in {time.time()-start_time:.2f} seconds.")
    