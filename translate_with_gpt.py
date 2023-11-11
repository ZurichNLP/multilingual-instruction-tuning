#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Translate Alpaca Eval instructions using OpenAI LLM.

Example call:

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_de.json \
        --tgt_lang "German" \
        --src_key "instruction"
        
    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_ru.json \
        --tgt_lang "Russian" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_hi.json \
        --tgt_lang "Hindi" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_fr.json \
        --tgt_lang "French" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_zh.json \
        --tgt_lang "Mandarin Chinese" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_el.json \
        --tgt_lang "standard modern Greek" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_is.json \
        --tgt_lang "Icelandic" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_sv.json \
        --tgt_lang "Swedish" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_da.json \
        --tgt_lang "Danish" \
        --src_key "instruction"
        
    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_es.json \
        --tgt_lang "Spanish" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_ca.json \
        --tgt_lang "Catalan" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_no.json \
        --tgt_lang "Norwegian Bokmål" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/alpaca_eval_instructions_en.json \
        --output_file data/alpaca_eval_instructions_bg.json \
        --tgt_lang "Bulgarian" \
        --src_key "instruction"

    python translate_with_gpt.py \
        --input_file data/guanaco_train_mono_1k_en.json \
        --output_file data/guanaco_train_mono_1k_zh.json \
        --tgt_lang "Mandarin Chinese" \
        --src_key "text" \
        --dataset_type "guanaco" \
        --model_name "gpt-3.5-turbo"
    
    python translate_with_gpt.py \
        --input_file data/lima_train_en.json \
        --output_file data/lima_train_de.json \
        --tgt_lang "German" \
        --src_key "text" \
        --dataset_type "lima" \
        --model_name "gpt-3.5-turbo-16k"

    python translate_with_gpt.py \
        --input_file data/lima_test_en.json \
        --output_file data/lima_test_hi.json \
        --tgt_lang "Hindi" \
        --src_key "text" \
        --dataset_type "lima" \
        --model_name "gpt-3.5-turbo-16k"
        
    python translate_with_gpt.py \
        --input_file data/outputs/llama_2_7b_hf_de_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --output_file alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8-translated.jsonl \
        --tgt_lang "English" \
        --src_key "system" \
        --dataset_type "alpaca_eval" \
        --model_name "gpt-3.5-turbo-1106" --original_prompts data/alpaca_eval_instructions_en.json
"""

import sys
import json
import argparse
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import math
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import tiktoken

from helpers import logger
from api_secrets import OPENAI_API_KEY

system_prompt = "You are a helpful assistant."

human_template = """Translate the following text into {tgt_lang}. 
Keep the structure of the original text and preserve things like code and names. 
Please ensure that your response contains only the translated text. 
The translation must convey the same meaning as the original and be natural for native speakers with correct grammar and proper word choices. 
Your translation must also use exact terminology to provide accurate information even for the experts in the related fields.

Original: "{prompt}"

Translation into {tgt_lang}:"""

lima_human_template = """Translate the following conversation between a human and an AI assistant into {tgt_lang}. 
Keep the structure of the original text and preserve things like code, names and role labels (e.g. <S1>, <S2>).
Please ensure that your response contains only the translated text. 
The translation must convey the same meaning as the original and be natural for native speakers with correct grammar and proper word choices. 
Your translation must also use exact terminology to provide accurate information even for the experts in the related fields.

Original: 

{prompt}

Translation into {tgt_lang}:"""

guanaco_human_template = """Translate the following conversation between a human and an AI assistant into {tgt_lang}. 
Keep the structure of the original text and preserve things like code, names and role labels (e.g. <S1>, <S2>).
Please ensure that your response contains only the translated text. 
The translation must convey the same meaning as the original and be natural for native speakers with correct grammar and proper word choices. 
Your translation must also use exact terminology to provide accurate information even for the experts in the related fields.

Original: 

{prompt}

Translation into {tgt_lang}:"""

# costings per model in USD for 1k input and output tokens
costings = {
    "gpt-3.5-turbo": (0.0015, 0.0020), 
    "gpt-3.5-turbo-16k": (0.003, 0.004), 
    "gpt-4": (0.03, 0.06), # 8k context
    "gpt-3.5-turbo-1106": (0.0010, 0.0020),
}


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, required=False, default=None)
    ap.add_argument("--output_file", type=str, required=False, default=None)
    ap.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--src_key", type=str, default="instruction")
    ap.add_argument("--tgt_lang", type=str, default="German")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dataset_type", type=str, default="alpaca_eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    ap.add_argument("--original_prompts", default=None, type=str, help="Path to original prompts file if required, e.g. for Alpaca Eval: data/alpaca_eval_instructions_en.json")
    return ap.parse_args()

def prepare_prompt(prompt, tgt_lang, human_template, system_prompt):

    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=human_template.format(prompt=prompt, tgt_lang=tgt_lang))

    return [system_message, human_message]

def run_llm(prompts, model_name="gpt-3.5-turbo-1106", max_tokens=1000):
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
        # request_timeout=60,
        )

    responses = []
    for prompt in tqdm(prompts, total=len(prompts)):
        with get_openai_callback() as cb:
            response = model.generate([prompt])
            total_cost += cb.total_cost
            total_prompt_tokens += cb.prompt_tokens
            total_completion_tokens += cb.completion_tokens
            total_tokens += cb.total_tokens
            yield response.generations[0][0].text

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
    
        
if __name__ == "__main__":

    args = set_args()

    data = pd.read_json(args.input_file, lines=True)
    
    # infer output file name if not specified - only works for Alpaca Eval outputs
    output_file = args.output_file
    if not output_file:
        if args.dataset_type == "alpaca_eval_outputs":
            ext = Path(args.input_file).suffix
            output_file = args.input_file.replace(ext, f"-with_{args.tgt_lang[:2].lower()}{ext}")
            logger.info(f"Inferred output file name: {output_file}")
        else:
            raise ValueError("Please specify an output file name. We currently only infer output file names for --dataset_type==\"alpaca_eval_outputs\"")

    if Path(output_file).exists() and not args.force:
        raise ValueError(f"Output file {output_file} already exists. Please specify a different output file name or use --force to overwrite.")
    
    if args.dataset_type == "alpaca_eval_outputs":
        if args.original_prompts:
            # get the original prompts as well
            en_prompts = pd.read_json(args.original_prompts, lines=True).rename(columns={"instruction": "source_en"})
            # merge the original prompts with the data
            data = pd.concat([data, en_prompts], axis=1)
        else:
            raise ValueError("Please specify the path to the original prompts file with --original_prompts")

        if args.limit:
            # sample the data up to the limit
            data['id'] = data.index # persist original index
            data = data.sample(n=min(args.limit, len(data)), random_state=args.seed).reset_index(drop=True)

    data = data.to_dict(orient="records")
    
    instructions = [item[args.src_key] for item in data]
    if args.dataset_type == "alpaca_eval": # for translating Alpaca Eval instructions from English to other languages
        prompts = [prepare_prompt(i, args.tgt_lang, human_template, system_prompt) for i in instructions]
    
    elif args.dataset_type == "alpaca_eval_outputs": # for translating non-English outputs from Alpaca Eval to English
        prompts = [prepare_prompt(i, args.tgt_lang, human_template, system_prompt) for i in instructions]

    elif args.dataset_type == "lima":
        # replace original role labels with <S1> and <S2> for better translation
        instructions = [i.replace("### Human:", "<S1>").replace("### Assistant:", "<S2>") for i in instructions]
        prompts = [prepare_prompt(i, args.tgt_lang, lima_human_template, system_prompt) for i in instructions]
    
    elif args.dataset_type == "guanaco":
        # replace original role labels with <S1> and <S2> for better translation
        instructions = [i.replace("### Human:", "<S1>").replace("### Assistant:", "<S2>") for i in instructions]
        prompts = [prepare_prompt(i, args.tgt_lang, guanaco_human_template, system_prompt) for i in instructions]
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # estimate number of tokens
    tokenizer = tiktoken.encoding_for_model(args.model_name)
    prompt_lengths = [len(tokenizer.encode(prompt[0].content + '\n\n' + prompt[1].content)) for prompt in prompts]
    logger.info(f"Total tokens: {sum(prompt_lengths)}")
    logger.info(f'Average tokens per prompt: {sum(prompt_lengths)/len(prompt_lengths)}')
    logger.info(f'Max tokens per prompt: {max(prompt_lengths)}')
    logger.info(f'Min tokens per prompt: {min(prompt_lengths)}')
    logger.info(f'Rough estimated cost: {((sum(prompt_lengths) * 2) / 1000) * costings[args.model_name][1]}')

    if args.inspect_only:
        sys.exit()

    if args.debug:
        prompts = prompts[:5]

    logger.info(f"Running LLM {args.model_name} on {len(prompts)} prompts ...")

    logger.info(f"Example prompt: {prompts[0]}")

    c = 0
    with open(output_file, "w", encoding='utf8') as f:
        for line, translation in zip(data, run_llm(prompts=prompts, model_name=args.model_name, max_tokens=args.max_tokens)):
            if args.dataset_type == "alpaca_eval_outputs":
                line[f'{args.src_key}_{args.tgt_lang[:2].lower()}_{args.model_name}'] = translation
            else:
                line = {args.src_key: translation}
            f.write(f"{json.dumps(line, ensure_ascii=False)}\n")
            c += 1

    logger.info(f"Wrote {c} translations to {output_file}")