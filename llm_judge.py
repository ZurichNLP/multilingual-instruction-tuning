#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage:

    python llm_judge.py \
        --input_file data/outputs/llama_2_7b_hf_zh_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --limit 3
        --force

"""

import sys
import json
import argparse
import time
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import LLMChain
from langchain.callbacks import get_openai_callback

from helpers import logger, str2bool
from api_secrets import OPENAI_API_KEY

system_prompt = """Pretend you are an expert language evaluator."""

human_template = """Act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user's prompt displayed below. 

Think step-by-step.

First, is the response in the same language as the prompt? If not, the response fails.

Second, categorize the response as one of the following:

Fail: the response either does not meet the requirements of the prompt, is not in the same language as the prompt, or it is ungrammatical or non-sensical.
Pass: the response is grammatically correct and sufficient given the prompt, but it could be improved.
Excellent: the response is an excellent response to the prompt and may be considered informative, interesting, and correct.

Provide a brief justification for your categorization.

Return only a RFC8259 compliant JSON following this schema: 

[{{"judgement": str, "justification": str}}]

For example:

Prompt: Was ist die Hauptstadt von Frankreich?
Response: Paris is the capital of France.
Assessment: [{{"judgement": "Fail", "justification": "language"}}]

Prompt: Was ist die Hauptstadt von Frankreich?
Response: Die Hauptstadt von Frankreich ist Paris.
Assessment: [{{"judgement": "Pass", "justification": "grammatical, uninteresting"}}]

Prompt: Was ist die Hauptstadt von Frankreich?
Response: Die Hauptstadt Frankreichs ist Paris. Mit etwa 2,15 Millionen Einwohnern ist Paris die grösste Stadt Frankreichs.
Assessment: [{{"judgement": "Excellent", "justification": "informative, interesting"}}]

Prompt: {prompt}
Response: {response}
Assessment: 
"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, type=str, help="Path to input file")
    ap.add_argument("--output_file", required=False, type=str, help="Path to output file")
    ap.add_argument("--model_name", required=False, type=str, default="gpt-3.5-turbo", help="Name of the model to use")
    ap.add_argument("--max_tokens", required=False, type=int, default=1000, help="Max tokens to use")
    ap.add_argument("--verbose", required=False, type=str2bool, default=False, help="Verbose")
    ap.add_argument("--limit", required=False, type=int, default=-1, help="Limit number of items to evaluate")
    ap.add_argument("--seed", required=False, type=int, default=42, help="Random seed for sampling items")
    ap.add_argument("--src_key", required=False, type=str, default="source", help="Source key")
    ap.add_argument("--tgt_key", required=False, type=str, default="system", help="Target key")
    ap.add_argument("--sleep_time", required=False, type=int, default=6, help="Sleep time between requests (e.g. for GPT-4)")
    ap.add_argument("--force", required=False, type=str2bool, default=False, help="Overwrites existing outputs if found, otherwise skip.")
    return ap.parse_args()

def prepare_prompts(prompt, response, human_template=human_template, system_prompt=system_prompt):

    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=human_template.format(prompt=prompt, response=response))

    return [system_message, human_message]

def get_llm_judgements(prompts, model_name="gpt-3.5-turbo", max_tokens=1000, sleep_time=4):
    """Get a judgement from the LLM judge."""
    
    total_cost = 0
    total_tokens = 0

    model = ChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        # request_timeout=120,
        )

    logger.info(f"Model: {model_name}")
    for i, prompt in enumerate(prompts):
        
        if model_name == "gpt-4":
            time.sleep(sleep_time)
        # else:
        #     time.sleep(2.5)

        with get_openai_callback() as cb:
            response = model.generate([prompt])
            total_cost += cb.total_cost
            total_tokens += cb.total_tokens

            logger.info(f"Progress: {i+1}/{len(prompts)}, Total cost: {total_cost}, Total tokens: {total_tokens}")
        
        yield response.generations[0][0].text, cb.total_cost


def parse_string_to_json(string):
    """Parse a string to a JSON object."""
    try:
        json_object = json.loads(string)[0]
    except ValueError as e:
        logger.error(f"Error parsing string to JSON: {e}")
        json_object = None

    return json_object

def is_valid_response(d):
    
    if d is None:
        return False
    if not isinstance(d, dict):
        return False
    if d.get("judgement") not in ["Fail", "Pass", "Excellent"]:
        return False
    if d.get("justification") is None:
        return False
    
    return True

def strip_quotes(text):
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


if __name__ == "__main__":

    args = set_args()

    data = pd.read_json(args.input_file, lines=True)
    data['id'] = data.index+1
    
    if args.limit > 0:
        data = data.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_file.replace(".jsonl", f"-{args.model_name.replace('-', '_')}-l{len(data)}.llm_eval")
    logger.info(f"Output file: {output_file}")

    if Path(output_file).exists() and not args.force:
        logger.warning(f"Output file already exists. Use --force to overwrite.")
        sys.exit(1)

    query_prompts = [prepare_prompts(strip_quotes(p), strip_quotes(r)) for p, r in zip(data[args.src_key], data[args.tgt_key])]
    logger.info(f"Model: {args.model_name}")
    logger.info(f"# Prompts: {len(data)}")
    logger.info(f"Output file: {output_file}")

    data[f'{args.model_name}-judgement'], data[f'{args.model_name}-justification'] = None, None
    data[f'{args.model_name}-cost'] = None

    for i, (result, cost) in tqdm(enumerate(get_llm_judgements(query_prompts, model_name=args.model_name, max_tokens=args.max_tokens, sleep_time=args.sleep_time)), total=len(data)):
        # print(query_prompts[i])
        result = parse_string_to_json(result)
        data.loc[i, f'{args.model_name}-judgement'] = result.get("judgement")
        data.loc[i, f'{args.model_name}-justification'] = result.get("justification")
        data.loc[i, f'{args.model_name}-cost'] = cost

    data.to_json(output_file, orient='records', lines=True, force_ascii=False)

    if args.verbose:
        print(data[f'{args.model_name}-judgement'].value_counts())

    logger.info(f"Finished LLM judgement. Output file: {output_file}")