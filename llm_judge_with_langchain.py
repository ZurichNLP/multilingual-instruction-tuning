#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage:

    python llm_judge.py \
        --input_file data/outputs/llama_2_7b_hf_zh_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --limit 3
        --force

    python llm_judge.py \
        --input_file alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8-translated.jsonl \
        
        
"""

import sys
import json
import argparse
import time
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np

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

system_prompt = """You are an expert language evaluator."""

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

human_template_likert = """You are evaluating a response that has been submitted for a particular task, using a specific set of standards. Below is the data: 

[BEGIN DATA]
***
[Task]: {prompt} 
***
[Submission]: {response}
***
[Criterion]: helpfulness: 

"1": "Not helpful - The generated text is completely irrelevant, unclear, or incomplete. It does not provide any useful information to the user." 

"2": "Somewhat helpful - The generated text has some relevance to the user's question, but it may be unclear or incomplete. It provides only partial information, logical inconsistencies, or the information provided may not be useful for the user's needs." 

"3": "Moderately helpful - The generated text is relevant to the user's question, and it provides a clear and complete answer. However, it may lack detail or explanation that would be helpful for the user." 

"4": "Helpful - The generated text is quite relevant to the user's question, and it provides a clear, complete, and detailed answer. It offers additional information or explanations that are useful for the user. However, some of the points of the response are somewhat repetitive or could be combined for greater clarity and concision" 

"5": "Very helpful - The generated text is highly relevant to the user's question, and it provides a clear, complete, and detailed answer. It offers additional information, explanations, or analogies that are not only useful but also insightful and valuable to the user. However, the structured of the response is not well-organized and there is no clear progression or logical sequence of different points in the response." 

"6": "Highly helpful - The generated text provides a clear, complete, and detailed answer. It offers additional information or explanations that are not only useful but also insightful and valuable to the user. The response is also in a logical and easy-to-follow manner by explicitly using headings, bullet points, or numbered lists to break up the information and make it easier to read." 
***
[END DATA]

Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. 
Provide your response as a RFC8259 compliant JSON following this schema:

{{"reasoning": str, "score": int}}

"""
# Then print the choice only from "1, 2, 3, 4, 5, 6" (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the selected choice again by itself on a new line.

# costings per model in USD for 1k input and output tokens
costings = {
    "gpt-3.5-turbo": (0.0015, 0.0020), 
    "gpt-3.5-turbo-16k": (0.003, 0.004), 
    "gpt-4": (0.03, 0.06), # 8k context
    "gpt-3.5-turbo-1106": (0.0010, 0.0020),
}


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, type=str, help="Path to input file")
    ap.add_argument("--output_file", required=False, type=str, help="Path to output file")
    ap.add_argument("--model_name", required=False, type=str, default="gpt-3.5-turbo", help="Name of the model to use")
    ap.add_argument("--max_tokens", required=False, type=int, default=1000, help="Max tokens to use")
    ap.add_argument("--eval_type", required=False, type=str, default="abs", help="Evaluation type (abs|likert)")
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
    total_prompt_tokens = 0
    total_completion_tokens = 0

    model = ChatOpenAI(
        model_name=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        request_timeout=60,
        )

    logger.info(f"Model: {model_name}")
    for i, prompt in enumerate(prompts):
        
        if model_name == "gpt-4":
            time.sleep(sleep_time)

        with get_openai_callback() as cb:
            response = model.generate([prompt])
            total_prompt_tokens += cb.prompt_tokens
            total_completion_tokens += cb.completion_tokens
            total_tokens += cb.total_tokens
            if cb.total_cost == 0.0:
                # calculate the cost manually, since langchain doesn't always include the cost of most recent models
                prompt_cost = cb.prompt_tokens/1000 * costings[model_name][0]
                completion_cost = cb.completion_tokens/1000 * costings[model_name][1]
                current_cost = prompt_cost + completion_cost
            else:
                current_cost = cb.total_cost
            total_cost += current_cost

            # logger.info(f"Progress: {i+1}/{len(prompts)}, Total cost: {total_cost}, Total tokens: {total_tokens}")
        
        yield response.generations[0][0].text, current_cost

    print(f"Model: {model_name}")
    print(f"# of prompts: {len(prompts)}")
    print(f"Total cost: {total_cost}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    # calculate total cost using costings dict since langchain doesn't include the cost of most recent models
    prompt_token_costs = total_prompt_tokens/1000 * costings[model_name][0]
    completion_token_costs = total_completion_tokens/1000 * costings[model_name][1]
    print(f"Cost for prompt tokens: {prompt_token_costs:.4f}")
    print(f"Cost for completion tokens: {completion_token_costs:.4f}")
    print(f"Total cost: {prompt_token_costs + completion_token_costs:.4f}")

def parse_string_to_json(string):
    """Parse a string to a JSON object."""
    try:
        json_object = json.loads(string)[0]
    except ValueError as e:
        logger.error(f"Error parsing string to JSON: {e}")
        json_object = None

    return json_object

def parse_likert_response(string):
    
    try:
        judgement = int(string.strip().split("\n")[-1])
    except ValueError as e:
        logger.error(f"Error parsing likert response from {string}: {e}")
        judgement = None
    
    try:
        justification = "\n".join(string.split("\n")[:-1])
    except ValueError as e:
        logger.error(f"Error parsing likert response from {string}: {e}")
        justification = None

    return {
        "judgement": judgement,
        "justification": justification,
    }


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
    
    # convert dataframe to dict
    data = data.to_dict(orient='records')

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_file.replace(".jsonl", f"-{args.model_name.replace('-', '_')}-l{len(data)}.{args.eval_type}_llm_eval")
    logger.info(f"Output file: {output_file}")

    if Path(output_file).exists() and not args.force:
        logger.warning(f"Output file already exists. Use --force to overwrite.")
        sys.exit(1)

    if args.eval_type == "abs":
        query_prompts = [prepare_prompts(strip_quotes(item[args.src_key]), strip_quotes(item[args.tgt_key])) for item in data]
    elif args.eval_type == "likert":
        query_prompts = [prepare_prompts(strip_quotes(item[args.src_key]), strip_quotes(item[args.tgt_key]), human_template=human_template_likert) for item in data]

    logger.info(f"Model: {args.model_name}")
    logger.info(f"# Prompts: {len(data)}")
    logger.info(f"Output file: {output_file}")

    with open(output_file, 'w', encoding='utf8') as outf:
        for item, (result, cost) in tqdm(zip(data, get_llm_judgements(query_prompts, model_name=args.model_name, max_tokens=args.max_tokens, sleep_time=args.sleep_time)), total=len(query_prompts)):

            if args.eval_type == "abs":
                result = parse_string_to_json(result)
                item[f'{args.model_name}-judgement'] = result.get("judgement")
                item[f'{args.model_name}-justification'] = result.get("justification")
            else:
                result = parse_string_to_json(result)
                item[f'{args.model_name}-score'] = result.get("score")
                item[f'{args.model_name}-reasoning'] = result.get("reasoning")

            item[f'{args.model_name}-cost'] = cost

            outf.write(f'{json.dumps(item, ensure_ascii=False)}\n')

    logger.info(f"Finished LLM judgement. Output file: {output_file}")