#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Generic translation script for translating a list of texts into a given target language using GPT-3.5-Turbo.

Example call:

    # for prompt translation (preparation of Alpaca Eval instructions for evaluation in multiple languages)
    python translate_with_gpt.py \
        --input_file data/alpaca_eval/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval/alpaca_eval_instructions_no.json \
        --tgt_lang "Norwegian" \
        --src_key "instruction" \
        --model_name "gpt-3.5-turbo-0613" \
        --dataset_type "alpaca_eval_prompts"

    # for translating generated outputs (e.g. from Alpaca Eval) into English for evaluation ablations
    python translate_with_gpt.py \
        --input_file data/alpaca_eval_outputs/llama_2_7b_hf_ml6_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --output_file data/alpaca_eval_outputs_translated/llama_2_7b_hf_ml6_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --tgt_lang "English" \
        --src_key "system" \
        --model_name "gpt-3.5-turbo-1106" \
        --original_prompts data/alpaca_eval_instructions_en.json --debug --force

        

"""

import sys
import argparse
from pathlib import Path
import time
import pandas as pd
import json
from tqdm import tqdm

import tiktoken

from helpers import logger
from api_request_parallel_processor import run_api_request_parallel_process

system_message = """You are a helpful assistant."""

user_message = """Translate the following text into {tgt_lang}. 
Keep the structure of the original text and preserve things like code and names. 
Please ensure that your response contains only the translated text. 
The translation must convey the same meaning as the original and be natural for native speakers with correct grammar and proper word choices. 
Your translation must also use exact terminology to provide accurate information even for the experts in the related fields.

Original: "{prompt}"

Translation into {tgt_lang}:"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, required=True)
    ap.add_argument("--output_file", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    ap.add_argument("--src_key", type=str, default="instruction")
    ap.add_argument("--tgt_lang", type=str, default="German")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--api_seed", type=int, default=42)
    ap.add_argument("--max_parallel_calls", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    ap.add_argument("--dataset_type", type=str, default="alpaca_eval_outputs", choices=["alpaca_eval_outputs", "alpaca_eval_prompts"], help="'outputs' refers to the generated responses which we translate for direct vs. translation evaluation ablations. 'Prompts' refers to the original prompts that we translate into multiple target languages for evaluations.")
    ap.add_argument("--original_prompts", default=None, type=str, help="Path to original prompts file if required, e.g. for Alpaca Eval: data/alpaca_eval_instructions_en.json")

    return ap.parse_args()

def prepare_prompt(prompt, tgt_lang, system_message=system_message, user_message=user_message):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message.format(prompt=prompt, tgt_lang=tgt_lang)},
    ]
        
if __name__ == "__main__":

    args = set_args()

    data = pd.read_json(args.input_file, lines=True)
    
    output_file = args.output_file
        
    if Path(output_file).exists() and not args.force:
        logger.error(f"Output file already exists. Use --force to overwrite.")
        sys.exit(0)
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to {output_file}")

    if args.dataset_type == "alpaca_eval_outputs":
        if args.original_prompts:
            # get the original prompts as well
            en_prompts = pd.read_json(args.original_prompts, lines=True).rename(columns={"instruction": "source_en"})
            # merge the original prompts with the data
            data = pd.concat([data, en_prompts], axis=1)
        else:
            raise ValueError("Please specify the path to the original prompts file with --original_prompts")
    
    if 'id' not in data.columns:
        data['id'] = data.index # persist original index
    
    if args.limit > 0:
            # sample the data up to the limit        
            data = data.sample(n=min(args.limit, len(data)), random_state=args.data_seed).reset_index(drop=True)

    data = data.to_dict(orient="records")
    
    texts = [item[args.src_key] for item in data]
    
    prompts = [prepare_prompt(text, args.tgt_lang, system_message, user_message) for text in texts]

    # estimate number of tokens
    tokenizer = tiktoken.encoding_for_model(args.model_name)

    prompt_lengths = [len(tokenizer.encode(prompt[0]['content'] + '\n\n' + prompt[1]['content'])) for prompt in prompts]
    logger.info(f"Total tokens: {sum(prompt_lengths)}")
    logger.info(f'Average tokens per prompt: {sum(prompt_lengths)/len(prompt_lengths)}')
    logger.info(f'Max tokens per prompt: {max(prompt_lengths)}')
    logger.info(f'Min tokens per prompt: {min(prompt_lengths)}')
    # logger.info(f'Rough estimated cost: {((sum(prompt_lengths) * 2) / 1000) * costings[args.model_name][1]}')

    if args.inspect_only:
        sys.exit()

    if args.debug:
        data = data[:5]
        prompts = prompts[:5]

    logger.info(f"Running LLM {args.model_name} on {len(prompts)} prompts ...")

    logger.info(f"Example prompt: {prompts[0]}")

    # run api requests in parallel -> results is a list of dicts containing 'content', 'system_fingerprint', 'prompt_tokens', 'completion_tokens', 'cost', 'model_name'
    results = run_api_request_parallel_process(
        args.model_name, 
        prompts, 
        max_parallel_calls=args.max_parallel_calls, 
        timeout=args.timeout, 
        seed=args.api_seed, 
        temperature=args.temperature, 
        expects_json=False
        )
    
                    
    # check that results match data
    assert len(data) == len(results), f"Length of data ({len(data)}) and results ({len(results)}) do not match!"


    c = 0
    with open(output_file, "w", encoding="utf8") as outf:
        for item, result in zip(data, results):
            if args.dataset_type == "alpaca_eval_prompts":
                item['instruction'] = result['content']
                del item['id']

            elif args.dataset_type == "alpaca_eval_outputs":
                item[f'{args.src_key}_en'] = result['content']
                item['translation_meta'] = {
                    "system_fingerprint": result['system_fingerprint'],
                    "prompt_tokens": result['prompt_tokens'],
                    "completion_tokens": result['completion_tokens'],
                    "cost": result['cost'],
                    "model_name": result['model_name'],
                    }
            
            outf.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            
            c += 1

    logger.info(f"Wrote {c} items to {output_file}")
    