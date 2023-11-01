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
        
"""

import json
import argparse
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
import tiktoken

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


costings = {
    "gpt-3.5-turbo": 0.002, # $0.002 per output token (input tokens are slightly cheaper)
    "gpt-3.5-turbo-16k": 0.004,
    "gpt-4": 0.06, # 8k context
}


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, required=False, default=None)
    ap.add_argument("--output_file", type=str, required=False, default=None)
    ap.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--src_key", type=str, default="instruction")
    ap.add_argument("--tgt_lang", type=str, default="de")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--inspect_only", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--dataset_type", type=str, default="alpaca_eval")
    return ap.parse_args()

def prepare_prompt(prompt, tgt_lang, human_template, system_prompt):

    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=human_template.format(prompt=prompt, tgt_lang=tgt_lang))

    return [system_message, human_message]

def run_llm(prompts, model_name="gpt-3.5-turbo", max_tokens=1000):
    """Get a judgement from the LLM judge."""
    
    total_cost = 0
    total_tokens = 0

    model = ChatOpenAI(
        model_name=model_name,
        temperature=0.01,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        request_timeout=60,
        )

    responses = []
    for prompt in tqdm(prompts, total=len(prompts)):
        with get_openai_callback() as cb:
            response = model.generate([prompt])
            total_cost += cb.total_cost
            total_tokens += cb.total_tokens
            yield response.generations[0][0].text

    print(f"Model: {model_name}")
    print(f"# of prompts: {len(prompts)}")
    print(f"Total cost: {total_cost}")
    print(f"Total tokens: {total_tokens}")
    
        
if __name__ == "__main__":

    args = set_args()

    data = pd.read_json(args.input_file, lines=True)

    instructions = data[args.src_key].tolist()
    
    if args.dataset_type == "alpaca_eval":
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
    print(f"Total tokens: {sum(prompt_lengths)}")
    print(f'Average tokens per prompt: {sum(prompt_lengths)/len(prompt_lengths)}')
    print(f'Max tokens per prompt: {max(prompt_lengths)}')
    print(f'Min tokens per prompt: {min(prompt_lengths)}')
    print(f'Rough estimated cost: {((sum(prompt_lengths) * 2) / 1000) * costings[args.model_name]}')

    if args.inspect_only:
        exit()

    if args.debug:
        prompts = prompts[:5]

    print(f"Running LLM {args.model_name} on {len(prompts)} prompts ...")

    print(f"Example prompt: {prompts[0]}")

    c = 0
    with open(args.output_file, "w", encoding='utf8') as f:
        for translation in run_llm(prompts=prompts, model_name=args.model_name, max_tokens=args.max_tokens):
            c += 1
            line = {args.src_key: translation}
            f.write(f"{json.dumps(line, ensure_ascii=False)}\n")

    print(f"Wrote {c} translations to {args.output_file}")