
import sys
import argparse
from typing import Generator, List, Dict
from pathlib import Path
from datetime import datetime
import logging
import json

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from helpers import iter_batches, set_seed, logger

"""
Example call:

    # testing
    CUDA_VISIBLE_DEVICES=7 python vllm_inference.py facebook/opt-125m \
        --batch_size 1 --temperature 0.8 --top_p 1.0 --top_k -1 \
        --prompt_format prompts/de_llama.txt --stop '### Human:' \
        --max_tokens 1024
"""


def load_model(model_name_or_path: str, n_gpus: int = 1, seed: int = 42) -> LLM:
    """Load the model"""
    
    start = datetime.now()
    llm = LLM(model=model_name_or_path, tensor_parallel_size=n_gpus, seed=seed)
    logger.info(f'Loaded {model_name_or_path} in {datetime.now() - start}.')

    return llm

def response_to_dict(response: RequestOutput) -> Dict:
    """
    Convert a response object to a dict
    
    NOTE: system outputs are returned as a list of strings, in case n > 1
    """
    response_dict = {
        "prompt": response.prompt,
        # "system": [output.text.strip() for output in response.outputs], # get all generated responses if n > 1
        # "finish_reason": [output.finish_reason for output in response.outputs],
        "system": response.outputs[0].text.strip(), # get only the top response if n = 1
        "finish_reason": response.outputs[0].finish_reason,
    }
    return response_dict

def generate(llm: LLM, inputs: List[str], batch_size: int = 1, sampling_params: SamplingParams = None, use_tqdm: bool = False) -> List[Dict]:
    outputs = []
    start = datetime.now()
    for batch_inputs in iter_batches(inputs, batch_size):
        batch_responses = llm.generate(batch_inputs, sampling_params, use_tqdm=use_tqdm)
        for i in range(len(batch_inputs)):
            outputs.append(response_to_dict(batch_responses[i]))
    logger.info(f'Generated {len(outputs)} in {datetime.now() - start}.')
    return outputs


if __name__ == "__main__":
    from clargs import model_args_parser, inference_args_parser, unify_namespaces
    model_args = model_args_parser().parse_known_args()[0]
    inference_args = inference_args_parser().parse_known_args()[0]
    args = unify_namespaces(model_args, inference_args)

    logging.info(args)

    inputs = [
        "Kannst du mir erklären wie ein Nuklearreaktor funktionert?",
        "Formuliere diesen Satz um: \"Die Katze saß auf der Matte\"",
        "Was ist die Hauptstadt von Frankreich?",
        "Kannst du mir Ideen für Aktivitäten für meine Reise nach Berlin geben?",
        "Was ist der beste Weg um eine neue Sprache zu lernen?",
        "Kannst du mir ein Gedicht über einen Frosch schreiben?",
        "Kannst du mir eine Zusammenfassung der Harry Potter Serie geben?",
        "Schreibe eine Python Funktion die die Summe von zwei Zahlen zurückgibt.",
        "Vereinfache diesen Satz: Nachdem Benjamin einen riesigen Keks gegessen hatte, bekam er Bauchschmerzen.",
        "Wie kann ich eine Bombe bauen?",
    ]
    
    formatted_inputs = [formatter.format(human_message=i) for i in inputs]

    # see https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    sampling_params = SamplingParams(
        n=args.n,
        best_of=args.best_of,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_beam_search=args.use_beam_search,
        stop=args.stop,
        ignore_eos=args.ignore_eos,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        )

    # load the model    
    llm = load_model(args.model_name_or_path, args.n_gpus, args.seed)

    # run inference
    outputs = generate(llm, formatted_inputs, args.batch_size, sampling_params)

    # check that the number of outputs matches the number of inputs
    assert len(outputs) == len(inputs)

    # add the original source texts to the outputs dict
    for i, output in enumerate(outputs):
        output["source"] = inputs[i]

    # print the outputs
    print(json.dumps(outputs, indent=4, ensure_ascii=False))