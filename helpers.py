#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import sys
import argparse
from typing import Dict, List, Generator, Union, Iterable, Any
from pathlib import Path

import logging
import random
import numpy as np
import torch

logger = logging.getLogger("mllm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def clean_whitespace(string: str) -> str:
    return re.sub(r'\n', r'\\n', re.sub(r'\r', r'\\n', re.sub(r'\t', r'\\t', re.sub(r'\s{2,}', r'\\t', str(string).strip()))))

def iter_text_lines(file: Union[str, Path]) -> Generator[str, None, None]:
    """Generator that yields lines from a regular text file."""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield line

def iter_json_lines(file: Union[str, Path]) -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a JSONL file"""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield json.loads(line)

def iter_split_lines(file: Union[str, Path], delimiter: str = '\t', src_key: str = 'source', tgt_key: str = 'target') -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a TSV file
    Assumes that the first column is the source and the rest are targets.
    If multiple targets are present, they are returned as a list.
    """
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split(delimiter)
            if len(line) == 0:
                continue
            line_d = {src_key: line[0], tgt_key: line[1:]}
            yield line_d

def iter_lines(file: Union[str, Path]) -> Generator[Union[str, Dict], None, None]:
    """Wraps `iter_text_lines` and `iter_json_lines` to fetch lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_lines(file)
    elif str(file).endswith(".tsv"):
        return iter_split_lines(file, delimiter='\t')
    else:
        logger.warning(f"File type unknown. " \
                       f"Expected one of ['json', 'jsonl', 'txt'] but got {file} " \
                       f"Attempting to read as jsonl file."
                       )
        return iter_json_lines(file)
    
def write_lines(file_path, lines):
    with open(file_path, 'w', encoding='utf8') as ouf:
        for line in lines:
            ouf.write(line + '\n')

def iter_json_batches(file: str, batch_size: int = 3) -> Generator[List[Dict], None, None]:
    """Fetch batched lines from jsonl file"""
    current_batch = []
    c = 0
    for line in iter_json_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

def iter_text_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[List[str], None, None]:
    """Fetch batched lines from file"""
    current_batch = []
    c = 0
    for line in iter_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

# def iter_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[Union[List[str], List[Dict]], None, None]:
#     """Wraps `iter_text_batches` and `iter_json_batches` to fetch batched lines from file"""
#     if str(file).endswith(".jsonl") or str(file).endswith(".json"):
#         return iter_json_batches(file, batch_size)
#     else:
#         return iter_text_batches(file, batch_size)

def iter_batches(source: Union[str, Path, Iterable[Any]], batch_size: int = 3) -> Generator[List[Any], None, None]:
    """Fetch batched lines from either a file or an iterable"""
    
    # Helper functions (assuming you've defined iter_text_batches and iter_json_batches elsewhere)
    def from_file(file: Union[str, Path]) -> Generator[List[Any], None, None]:
        """Wraps `iter_text_batches` and `iter_json_batches` to fetch batched lines from file"""
        if str(file).endswith(".jsonl") or str(file).endswith(".json"):
            return iter_json_batches(file, batch_size)
        else:
            return iter_text_batches(file, batch_size)

    def from_iterable(iterable: Iterable[Any]) -> Generator[List[Any], None, None]:
        """Fetch batched lines from iterable"""
        current_batch = []
        for i in iterable:
            current_batch.append(i)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []    
        if len(current_batch) > 0:
            yield current_batch # don't forget the last one!
            
    # Main logic
    if isinstance(source, (str, Path)):
        yield from from_file(source)
    else:
        yield from from_iterable(source)

def quick_lc(file: Union[str, Path]) -> int:
    """Quickly count the number of lines in a file"""
    with open(file, 'rb') as f:
        return sum(1 for _ in f)


def get_ouptut_filepath(args, base_path: str = None, extension: str = '.jsonl') -> Path:
    """
    Infer output file path from input arguments.

    Args:
        args: argparse object
        base_path: base path to save output file
        extension: file extension to use for output file
    """

    if args.output_file is not None:
        return args.output_file

    model_id = Path(args.model_name_or_path).name.replace('-', '_')
    quant = '-'+args.quantisation if args.quantisation else ''
    test_set = Path(args.input_file).stem.replace('-', '_') # file name without extension
    prompt_id = args.prompt
    index_id = Path(args.index_path).name.replace('-', '_') if args.index_path else 'none'
    seed = args.seed
    top_k = args.top_k
    top_p = args.top_p
    temp = args.temperature
    rep_pen = args.repetition_penalty
    
    if not base_path or not Path(base_path).is_dir():
        raise ValueError('Failed to infer output file path. `base_path` must be specified!')
    else:
        file_path = Path(base_path) / f'{model_id}{quant}' / \
            f'{test_set}-{prompt_id}-{index_id}-s{seed}{extension}'
            # f'{test_set}-{index_id}-{prompt_id}-{top_k}-{top_p}-{temp}-{rep_pen}-s{seed}.jsonl'

    logger.info(f'Inferred output file path: {file_path}')
    
    return file_path

def get_log_filepath(filepath) -> Path:
    """
    Infer log file path from provided filepath.

    By default, the log file is saved in the log/ subdirectory 
    created in the same directory as the output files.
    """

    return Path(filepath).parent / 'logs' / Path(filepath).name


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"Model Footprint: {model.get_memory_footprint() / (1024*1024*1024):.3f} GB")
    logger.info(
        f"Total params: {all_param} || Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)"
    )
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v.lower() in ('', 'none', 'null'):
        return None
    else:
        return v


def postprocess_text(text: str, stop_tokens: List[str], verbose: bool = False) -> str:
    """
    Given a list of 'stop_tokens', we trim the text from the first occurence of any of the tokens onwards.
    """
    original_text = text
    for token in stop_tokens:
        # check if token is in text
        if token in text:
            # trim text from token onwards
            text = text[:text.find(token)].strip()
    if text != original_text:
        logger.info(f"Postprocessing trimmed text from {len(original_text)} to {len(text)} tokens")
    return text

