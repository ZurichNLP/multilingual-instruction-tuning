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

import sys
from typing import List, Dict
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import evaluate

import torch
from sacremoses import MosesPunctNormalizer, MosesTokenizer

from helpers import logger
from open_lid import LIDModel

mpn = MosesPunctNormalizer()
bleu = evaluate.load('sacrebleu')
chrf = evaluate.load('chrf')
# comet = evaluate.load('comet')
perplexity = evaluate.load('perplexity', module_type='metric')

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_file', type=str, help="File to evaluate. Expects a jsonl file with 'source', 'reference' and 'system' keys")
    ap.add_argument('-o', '--output_file', type=str, default=None, help='')    
    ap.add_argument('-v', '--verbose', action='store_true', help='if provided, print verbose output.')
    ap.add_argument('--use_cuda', action='store_true', help='if provided, compute GPU-based metrics.')
    ap.add_argument('--lang', type=str, default='en', help='target language. Required for bertscore and tokenization')
    ap.add_argument('--src_key', type=str, default='source', help='key for source texts in jsonl file')
    ap.add_argument('--ref_key', type=str, default='reference', help='key for reference texts in jsonl file')
    ap.add_argument('--tgt_key', type=str, default='system', help='key for system texts in jsonl file')
    ap.add_argument('--fasttext_model_path', type=str, default=None, help='path to fasttext language detection model')
    ap.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='tokens to remove from system texts before computing metrics')
    ap.add_argument('--non_answer_str', type=str, default=None, help='string to use as no answer found')
    ap.add_argument('--force', action='store_true', help='if provided, overwrite existing output files.')
    return ap.parse_args()

def tokenize_texts(texts: List[str], lang: str = 'en') -> List[str]:
    """
    Tokenize texts for metrics in Hugging Face evaluation package.

    :texts: are expected to be either a list of lists of strings (where each list of strings is a reference)
    or a list of strings (where each string is a sys output).
    """
    mt = MosesTokenizer(lang=lang)
    return [mt.tokenize(mpn.normalize(text), return_str=True) for text in texts]

def compute_bleu(
    predictions: List[str], 
    references: List[List[str]], 
    lang: str = 'en',
    ) -> Dict:
    """
    https://huggingface.co/spaces/evaluate-metric/sacrebleu
    
    predictions = ["hello there", "general kenobi"]
    references = [
        ["hello there general kenobi", "hello there !"],
        ["foo bar foobar"]
    ]   
    """ 
    
    # if kwargs.get('verbose'):
    logger.info(f'Computing BLEU with {len(predictions)} predictions and {len(references)} references...')
    
    # sacrebleu applies tokenization to the predictions and references separately, 
    # but can override this behavior by passing setting force=True
    if lang == 'zh':
        tokenize = 'zh'
    else:
        tokenize = '13a'

    return bleu.compute(predictions=predictions, references=references, tokenize=tokenize)

def compute_chrf(
    predictions: List[str],
    references: List[List[str]],
    lang: str = 'en',
    ) -> Dict:
    """
    """

    # if kwargs.get('verbose'):
    logger.info(f'Computing chrF with {len(predictions)} predictions and {len(references)} references...')
    
    try:
        predictions = tokenize_texts(predictions, lang=lang)
        references = tokenize_texts(references, lang=lang)
    except AssertionError:
        logger.warning(f'Failed to tokenize texts. Computing chrF without tokenization.')
    
    return chrf.compute(predictions=predictions, references=references, 
                        char_order=2, beta=2
                        )

def compute_perplexity(
    predictions: List[str], 
    model_id: str = 'distilgpt2', 
    batch_size: int = 8, 
    max_length: int = 1024,
    lang: str = 'en',
    **kwargs
    ):
    """
    https://huggingface.co/spaces/evaluate-metric/perplexity

    input_texts = ["hello there", "general kenobi"]
    """
    if lang != 'en':
        model_id = 'ai-forever/mGPT'
    else:
        model_id = 'distilgpt2'
    
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
    
    return ppl_scores['mean_perplexity'], model_id     

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

    if Path(args.output_file).exists() and not args.force:
        logger.error(f"Output file {args.output_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    # start timer
    logger.info("Starting evaluation...")
    start_time = time.time()

    data = pd.read_json(args.input_file, orient='records', lines=True)

    logger.info(f"Read {len(data[args.src_key])} lines from {args.input_file}")
    
    if 'alpaca_eval_instructions' in args.input_file and len(data) != 805:
        raise ValueError(
            f"Wrong number of samples in input file. "
            f"Expected 805 samples in {args.input_file}, but got {len(data)}."
            )

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
        # # if we have multiple references per sample, we need to transpose the list of lists
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
        metrics['ppl'], metrics['ppl_model'] = compute_perplexity(sys_sents, lang=args.lang)
    else:
        metrics['ppl'], metrics['ppl_model'] = None, None

    if refs_sents is not None:
        # compute BLEU, chrF
        # postprocess system outputs to take only the first sentence upto linebreak
        sys_sents = [s.split('\n')[0] for s in sys_sents]
        # metrics.update(mt_metrics.compute(predictions=sys_sents, references=refs_sents, sources=src_sents))
        metrics['bleu'] = compute_bleu(predictions=sys_sents, references=refs_sents, lang=args.lang)['score']
        metrics['chrf'] = compute_chrf(sys_sents, refs_sents, lang=args.lang)['score']

    # add filename
    metrics['n'] = len(sys_sents)
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