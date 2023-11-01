#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# from root dir of repo
CUDA_VISIBLE_DEVICES=7 python -m llm_dqa.inference.inference llm_dqa/resources/models/llama-2-7b-hf_16bit_guanaco_dpr_quad_1024_merged/ \
    --input_file llm_dqa/resources/data/basic_questions.jsonl \
    --batch_size 1 \
    --output_path llm_dqa/resources/outputs/ \
    --prompt_format llm_dqa/inference/prompts/de_llama \
    --src_key question \
    --stop '### Human:' '\n### Human:'

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m llm_dqa.inference.inference llm_dqa/resources/models/llama-2-70b-hf_16bit_guanaco_dpr_quad_1024_merged/ \
    --input_file llm_dqa/resources/data/fest1/OAFF23_faq-e5.jsonl \
    --batch_size 1 \
    --output_path llm_dqa/resources/outputs/ \
    --prompt_format llm_dqa/inference/prompts/de_llama \
    --src_key question --tgt_key answer --ctx_key contexts \
    --stop '### Human:' '\n### Human:' --n_gpus 4

# no contexts
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m llm_dqa.inference.inference llm_dqa/resources/models/llama-2-70b-hf_16bit_guanaco_dpr_quad_1024_merged/ \
    --input_file llm_dqa/resources/data/fest1/OAFF23_faq.jsonl \
    --batch_size 1 \
    --output_path llm_dqa/resources/outputs/ \
    --prompt_format llm_dqa/inference/prompts/de_llama \
    --src_key question --tgt_key answer \
    --stop '### Human:' '\n### Human:' --n_gpus 4

    
"""

from typing import List, Optional
import json
from pathlib import Path
from tqdm import tqdm
import time
from jinja2 import Environment, FileSystemLoader

from clargs import (
    model_args_parser, 
    inference_args_parser, 
    data_args_parser, 
    unify_namespaces,
)
from vllm_inference import *

from helpers import logger, set_seed, iter_batches, quick_lc, postprocess_text


def get_ouptut_filepath(args, base_path: str = None, extension: str = '.jsonl') -> Path:
    """
    Infer output file path from input arguments.

    Args:
        args: argparse object
        base_path: base path to save output file
        extension: file extension to use for output file
    """

    model_id = Path(args.model_name_or_path).name.replace('-', '_')
    test_set = Path(args.input_file).stem.replace('-', '_') # file name without extension
    prompt_format = Path(args.prompt_format).stem.replace('-', '_') if args.prompt_format else 'none'
    index_id = Path(args.index_path).name.replace('-', '_') if args.index_path else 'none'
    seed = args.seed
    top_k = 0 if args.top_k < 0 else args.top_k
    top_p = args.top_p
    temp = args.temperature
    batch_size = args.batch_size
    # rep_pen = args.repetition_penalty

    file_path = Path(model_id) / f'{test_set}-{index_id}-{prompt_format}-s{seed}-k{top_k}-p{top_p}-t{temp}-b{batch_size}{extension}'
    
    if base_path is not None:
        file_path = Path(base_path) / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'Inferred output file path: {file_path}')
    
    return file_path


def truncate_text(input_text, tokenizer, max_input_length, truncate_from_start: bool = True):
    """Handle text truncation."""
    overflow = 0
    while True:
        tokens = tokenizer.encode(input_text)
        if len(tokens) <= max_input_length:
            if overflow > 0:
                logger.warning(f"Input sequence exceeded max_input_length ({max_input_length}). " \
                                f"truncated {overflow} words from the input text.")
            return input_text
        else:
            # remove the last word from the text
            if truncate_from_start:
                input_text = ' '.join(input_text.split(' ')[1:])
            else:
                input_text = ' '.join(input_text.split(' ')[:-1])
            overflow += 1
    
    
def prepare_inputs_for_generation(
        input_texts: List[str], 
        prompt_template: Environment = None,
        tokenizer = None,
        max_input_length: int = 1024,
        truncate_from_start: bool = True,
        ) -> List[str]:
    """
    Convert input_texts and context_texts (optional) into prompted_input_texts, which are formatted according to the prompt.

    """

    # strip leading and trailing quotes from input texts if present (artifact from gpt3-translation format)
    input_texts_ = []
    for i in input_texts:
        if i[0] == '"' and i[-1] == '"':
            input_texts_.append(i[1:-1]) # remove leading and trailing quotes
        else:
            input_texts_.append(i) # do nothing

    prompted_input_texts = [prompt_template.render(instruction=input_texts[i]) for i in range(len(input_texts_))]

    prompted_input_texts = [truncate_text(i, tokenizer, max_input_length, truncate_from_start) for i in prompted_input_texts]

    return prompted_input_texts


def main(args):
    logger.info(args)

    if args.output_file is not None:
        output_file = args.output_file
    else: # infer output file from args provided
        output_file = get_ouptut_filepath(args, args.output_path, '.jsonl')
    
    # Create output directory if it does not exist
    if Path(output_file).exists() and not args.force:
        raise FileExistsError(f'Output file {output_file} already exists! Use --force to overwrite. Skipping inference run...')
    elif Path(output_file).exists() and args.force:
        logger.warning(f'Output file {output_file} already exists and will be overwritten!')
        Path(output_file).unlink()
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Write arguments to file
    with open(Path(output_file).with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
        logger.info(f'Arguments written to {Path(output_file).with_suffix(".json")}')
    
    prompts_dir = Path(args.prompt_format).parent.absolute()
    prompt_name = Path(args.prompt_format).stem
    prompt_template = Environment(loader=FileSystemLoader(prompts_dir)).get_template(f"{prompt_name}.txt")

    # # see https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        best_of=args.best_of,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_beam_search=args.use_beam_search,
        stop=args.stop_tokens,
        ignore_eos=args.ignore_eos,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        )

    # load the model    
    llm = load_model(args.model_name_or_path, args.n_gpus, args.seed)
    
    # Read input file
    if args.limit > 0:
        lc = args.limit
    else:
        lc = quick_lc(args.input_file)

    c = 0
    
    for batch_lines in tqdm(iter_batches(args.input_file, args.batch_size), total=lc//args.batch_size):
        input_texts = [line[args.src_key] for line in batch_lines]

        ref_texts = None
        if args.tgt_key and args.tgt_key in batch_lines[0]:
            ref_texts = [line[args.tgt_key] for line in batch_lines]
        
        # Write predictions to file
        with open(output_file, 'a', encoding='utf-8') as f:
            
            batch_inputs = prepare_inputs_for_generation(
                input_texts=input_texts, 
                prompt_template=prompt_template, 
                tokenizer=llm.get_tokenizer(),
                max_input_length=args.max_input_length,
                truncate_from_start=args.truncate_from_start,
                )

            if batch_inputs == []:
                raise ValueError("batch_inputs is empty!")
            
            if args.verbose:
                logger.info(f"Current batch (1/{len(batch_inputs)}): {batch_inputs[0]}")

            start_time = time.time()
            batch_outputs = generate(llm, batch_inputs, args.batch_size, sampling_params, use_tqdm=False)

            if len(batch_outputs) != len(batch_inputs):
                raise ValueError("batch_outputs and batch_inputs have different lengths!")
            
            end_time = time.time()
            # add the original source texts to the outputs dict
            for i, output_dict in enumerate(batch_outputs):
                output_dict["source"] = input_texts[i]
                # output_dict["contexts"] = context_texts[i] if context_texts else None
                output_dict["secs"] = ((end_time - start_time) / len(batch_outputs))
                if ref_texts:
                    output_dict["reference"] = ref_texts[i]
                if args.stop_tokens:
                    output_dict["system"] = postprocess_text(output_dict["system"], args.stop_tokens, args.verbose)
                # write the outputs to the output file
                f.write(f"{json.dumps(output_dict, ensure_ascii=False)}\n")
                if args.verbose:
                    logger.info(f"Time taken: {output_dict['secs']:.2f} seconds")
                    logger.info(f"Stopped on: {output_dict['finish_reason']}")
                    logger.info(f"Output: {output_dict['system']}")
            
        c += len(batch_outputs)
        if args.limit > 0 and c >= args.limit:
            logger.info(f'Limit of {args.limit} predictions reached! Exiting...')
            break
    
    logger.info(f"Results: {str(Path(output_file).absolute())}")

if __name__ == "__main__":
    model_args = model_args_parser().parse_known_args()[0]
    inference_args = inference_args_parser().parse_known_args()[0]
    data_args = data_args_parser().parse_known_args()[0]
    args = unify_namespaces(model_args, inference_args, data_args)
    main(args)