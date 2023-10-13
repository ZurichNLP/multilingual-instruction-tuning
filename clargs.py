#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from helpers import str2bool, str2none

def unify_namespaces(*namespaces):
    unified_dict = {}

    for ns in namespaces:
        ns_dict = vars(ns)
        
        # Check for overlapping keys and warn
        for key, value in ns_dict.items():
            if key in unified_dict and unified_dict[key] != value:
                print(f"Warning: Argument '{key}' from namespace {ns} is overriding previous value '{unified_dict[key]}' with '{value}'.")
            unified_dict[key] = value

    return argparse.Namespace(**unified_dict)

# model related args
def model_args_parser():

    ap = argparse.ArgumentParser(description='LLM model arguments')

    ap = argparse.ArgumentParser()

    # model loading params
    ap.add_argument("model_name_or_path", default="facebook/opt-125m", type=str, help="Model name or path")
    ap.add_argument("--n_gpus", default=1, type=int, help="Number of GPUs to use for inference")
    ap.add_argument("--seed", default=42, type=int, help="Random seed for initialization")

    return ap

# api related args
def api_args_parser():
    
    ap = argparse.ArgumentParser(description='api arguments')
    
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--prompt", type=str, default="San Francisco is a")
    ap.add_argument("--model_url", type=str, default="http://localhost:8000/generate")
    return ap

# inference related args
def inference_args_parser():

    ap = argparse.ArgumentParser(description='inference arguments')

    # inference params
    ap.add_argument("--num_return_sequences", default=1, type=int, help="Number of samples to generate for each prompt")
    ap.add_argument("--best_of", default=1, type=int, help="Number of samples to generate for each prompt")
    ap.add_argument("--presence_penalty", default=0.0, type=float, help="Presence penalty")
    ap.add_argument("--frequency_penalty", default=0.2, type=float, help="Frequency penalty")
    ap.add_argument("--temperature", default=0.8, type=float, help="Temperature")
    ap.add_argument("--top_p", default=0.9, type=float, help="Top p")
    ap.add_argument("--top_k", default=50, type=int, help="Top k")
    ap.add_argument("--use_beam_search", default=False, type=bool, help="Use beam search")
    ap.add_argument("--stop_tokens", nargs="+", default=None, type=str, help="Stop tokens")
    ap.add_argument("--ignore_eos", default=False, type=bool, help="Ignore EOS")
    ap.add_argument("--max_tokens", default=1024, type=int, help="Max output tokens before forcing EOS")
    ap.add_argument("--logprobs", default=None, type=str, help="Logprobs")
    ap.add_argument("--batch_size", default=1, type=int, help="Batch size")
    
    return ap

def data_args_parser():

    ap = argparse.ArgumentParser(description='data arguments')

    ap.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file for inference",
    )

    ap.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for inference",
    )

    ap.add_argument(
        "--output_path",
        type=str,
        default="llm_dqa/resources/outputs/",
        help="Output path for inference. Full file path will be inferred from args",
    )

    ap.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Log path for inference. If not specified, logs are written to models' outputs' subdir.",
    )

    # expected key in the input file
    ap.add_argument(
        "--src_key",
        type=str,
        default="question", # QuestionText for MSQA
        help="Source column for inference",
    )

    ap.add_argument(
        "--tgt_key",
        type=str,
        default="answer", #  ProcessedAnswerText for MSQA
        help="Target column for inference",
    )

    ap.add_argument(
        "--ctx_key",
        type=str,
        default=None,
        help="Context column for inference",
    )

    ap.add_argument(
        "--instruction_prefix",
        type=str,
        default=None,
        help="Instruction prefix for inference (experimental)",
    )

    ap.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to index for RAG model",
    )

    ap.add_argument("--prompt_format", default="prompts/dummy", type=str, help="Prompt format template")

    ap.add_argument("--verbose", action="store_true", help="Verbose")

    ap.add_argument("--limit", default=-1, type=int, help="Limit number of examples to process")

    ap.add_argument(
        "--truncate_from_start", 
        type=str2bool, 
        nargs="?", 
        const=True, 
        default=True, 
        help="When handling model inputs that exceed model size, truncate from start instead of end"
    )
    
    ap.add_argument(
        "--max_input_length", 
        default=4096, 
        type=int, 
        help="Max input length that the model accepts"
    )


    return ap

# retrieval related args
def retrieval_args_parser():
    
    ap = argparse.ArgumentParser(description='retrieval arguments')

    ap.add_argument(
        "-k",
        "--k",
        type=int,
        default=3,
        help="Number of retrieved contexts",
    )

    ap.add_argument(
        "--fetch_k",
        type=int,
        default=3,
        help="Number of retrieved documents before applying filtering (must be >= k)",
    )

    return ap


if __name__ == "__main__":

    model_args = model_args_parser().parse_known_args()[0]
    print(model_args)

    data_args = data_args_parser().parse_known_args()[0]
    print(data_args)

    inference_args = inference_args_parser().parse_known_args()[0]
    print(inference_args)

    retrieval_args = retrieval_args_parser().parse_known_args()[0]
    print(retrieval_args)

    args = unify_namespaces(model_args, data_args, inference_args, retrieval_args)
    print(args)