#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:

    python -m evaluation \
        data/outputs/llama_2_7b_hf_ml1_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s0-k50-p0.9-t0.8-b8.jsonl \
        --lang de \
        --src_key source \
        --tgt_key system

"""

from typing import List, Dict
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import evaluate

import torch

# from llm_dqa.utils.helpers import str2bool, postprocess_text, logger
# from llm_dqa.api_secrets import OPENAI_API_KEY
# set OpenAI API key as environment variable (required for ragas)
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from helpers import str2bool, logger
from open_lid import LIDModel

perplexity = evaluate.load('perplexity', module_type='metric')

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_file', type=str, help="File to evaluate. Expects a jsonl file with 'source', 'reference' and 'system' keys")
    ap.add_argument('-o', '--output_file', type=str, default=None, help='')    
    ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', const=True, default=False, help='if provided, print verbose output.')
    ap.add_argument('--use_cuda', type=str2bool, nargs='?', const=True, default=False, help='if provided, compute GPU-based metrics.')
    ap.add_argument('--lang', type=str, default='en', help='target language. Required for bertscore and tokenization')
    ap.add_argument('--src_key', type=str, default='source', help='key for source texts in jsonl file')
    ap.add_argument('--ref_key', type=str, default='reference', help='key for reference texts in jsonl file')
    ap.add_argument('--tgt_key', type=str, default='system', help='key for system texts in jsonl file')
    ap.add_argument('--fasttext_model_path', type=str, default=None, help='path to fasttext language detection model')
    ap.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='tokens to remove from system texts before computing metrics')
    ap.add_argument('--non_answer_str', type=str, default=None, help='string to use as no answer found')
    return ap.parse_args()


def compute_perplexity(
    predictions: List[str], 
    model_id: str = 'distilgpt2', 
    batch_size: int = 8, 
    max_length: int = 1024,
    **kwargs
    ):
    """
    https://huggingface.co/spaces/evaluate-metric/perplexity

    input_texts = ["hello there", "general kenobi"]
    """

    model_id = 'ai-forever/mGPT'
    # lang = kwargs.get('lang', 'en')
    # if lang == 'en':
    #     model_id = 'distilgpt2'
    # elif lang == 'de':
    #     model_id = 'benjamin/gpt2-wechsel-german'
    # elif lang == 'ru':
    #     model_id = 'ai-forever/rugpt3small_based_on_gpt2'
    # else:
    #     raise ValueError(f"Language {lang} not supported.")
    
    # filter out empty strings
    predictions_ = [p for p in predictions if p.strip() != '']
    logger.info(f'Filtered out {len(predictions) - len(predictions_)} empty strings.')

    logger.info(f'Computing PPL on {len(predictions_)} input texts with {model_id} ...')

    while True:
        try:
            logger.info(f"Trying batch size {batch_size}...")
            ppl_scores = perplexity.compute(
                predictions=predictions_, 
                model_id=model_id,
                add_start_token=True,
                batch_size=batch_size,
                max_length=max_length,
                device='cuda'
            )
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OutOfMemoryError encountered. Reducing batch size from {batch_size} to {batch_size//2} and retrying.")
            batch_size //= 2

            if batch_size < 1:
                # raise ValueError("Batch size is too small to process. Cannot continue.")
                logger.warning("Failed to compute PPL!")
                return None
    
    return ppl_scores['mean_perplexity']      

def calculate_agreement(src_langs, tgt_langs):
    """calculates the proportion of positions at which the two lists have equal values, i.e. agreement"""
    if len(src_langs) != len(tgt_langs):
        raise ValueError("Length of system languages and source languages not equal")
    
    # Count the number of matching pairs
    matching_pairs = sum(1 for s, t in zip(src_langs, tgt_langs) if s == t)
    
    # Calculate the proportion of matching pairs
    proportion_of_matches = matching_pairs / len(src_langs)
    
    return proportion_of_matches

def main(args):

    # start timer
    logger.info("Starting evaluation...")
    start_time = time.time()

    data = pd.read_json(args.input_file, orient='records', lines=True)

    logger.info(f"Read {len(data[args.src_key])} lines from {args.input_file}")
    
    if args.src_key is not None and args.src_key in data.columns:
        src_sents = data[args.src_key].to_list()
    else:
        src_sents = None

    if args.tgt_key is not None and args.tgt_key in data.columns:
        sys_sents = data[args.tgt_key].to_list()
        # if we have multiple system outputs per sample, we select the first one
        if isinstance(sys_sents[0], list):
            sys_sents = [s[0] for s in sys_sents]
    else:
        sys_sents = None
 
    if args.ref_key is not None and args.ref_key in data.columns:
        refs_sents = data[args.ref_key].to_list()
        # if we have multiple references per sample, we need to transpose the list of lists
        if isinstance(refs_sents[0], list):
            refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs_per_sample] to [# refs_per_sample, # samples]
        else:
            refs_sents = [[ref] for ref in refs_sents]
    else:
        refs_sents = None
 
    # if args.stop_tokens is not None:
    #     sys_sents = [postprocess_text(s, args.stop_tokens, verbose=args.verbose) for s in sys_sents]

    # load LID model
    lid_model = LIDModel()

    # assign language to each text
    src_langs = [lid_model.predict(src_sent)[0] for src_sent in src_sents]
    sys_langs = [lid_model.predict(sys_sent)[0] for sys_sent in sys_sents]

    # language agreement between source and system texts
    metrics = {}
    
    metrics['lang_match'] = calculate_agreement(src_langs, sys_langs)

    metrics['tgt_lang'] = calculate_agreement([lid_model.get_long_tag(args.lang)]*len(sys_langs), sys_langs)

    if args.use_cuda:
        metrics['ppl'] = compute_perplexity(sys_sents, lang=args.lang)
    else:
        metrics['ppl'] = None
    
    # add filename
    metrics['file'] = args.input_file

    metrics = pd.DataFrame(metrics, index=[0]).round(3)
    
    if args.output_file is not None:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output_file, index=False)
        logger.info(f"Saved metrics to {args.output_file}")

    logger.info(f"Finished evaluation in {time.time() - start_time:.2f} seconds.")

    print(metrics.to_csv(index=False))

if __name__ == "__main__":
    args = set_args()
    main(args)