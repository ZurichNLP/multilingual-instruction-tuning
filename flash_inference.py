import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from peft import AutoPeftModelForCausalLM


# @dataclass
# class ScriptArguments:
#     """
#     These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
#     """
#     model_name_or_path: Optional[str] = field(
#         default="facebook/opt-125m",
#         metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."},
#     )
    
#     max_seq_length: Optional[int] = field(
#         default=1024,
#         metadata={"help": "The maximum sequence length that this model might ever be used with. Typically 512, 1024, 2048."},
#     )

class StopOnTokens(StoppingCriteria):

    def __init__(self, stop_token_ids: List[int], tokenizer: PreTrainedTokenizer):

        self.stop_token_ids = stop_token_ids
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-len(stop_id):].tolist() == stop_id:
                return True
        return False


def truncate_text(
        input_text: str, 
        tokenizer: PreTrainedTokenizer, 
        max_input_length: int = 1024,
        truncate_from_start: bool = False):
    """Handle text truncation."""
    overflow = 0
    while True:
        tokens = tokenizer.encode(input_text)
        if len(tokens) <= max_input_length:
            if overflow > 0:
                print(
                    f"Input sequence exceeded max_input_length ({max_input_length}). " \
                    f"truncated {overflow} words from the input text."
                    )
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
        tokenizer: PreTrainedTokenizer = None,
        max_input_length: int = 1024,
        truncate_from_start: bool = False,
        ) -> List[str]:
    """
    Convert input_texts into prompted_input_texts, which are formatted according to the prompt.
    """

    prompt = "### Human: {}### Assistant:"

    input_texts_ = []
    for input_text in input_texts:
        if input_text.startswith('"'):
            input_text = input_text[1:]
        if input_text.endswith('"'):
            input_text = input_text[:-1]
        input_texts_.append(input_text)

    input_texts = [truncate_text(i, tokenizer, max_input_length, truncate_from_start) for i in input_texts_]

    prompted_input_texts = [prompt.format(i) for i in input_texts]

    return prompted_input_texts


model_name_or_path = "/scratch/tannon/mllm/llama-2-7b-hf_mono/"
test_dataset = "data/alpaca_eval_instructions_en.json"
batch_size = 4
stop_tokens = ["### Human:", "### Assistant:", "## Human:", "## Assistant:", "# Human:", "# Assistant", " Human:", " Assistant:"]

# # load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=False,
    use_flash_attention_2=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load dataset from the hub and get a sample
dataset = load_dataset("json", data_files=test_dataset)['train']

debug = True
if debug:
    dataset = dataset.select(range(10))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

stop_token_ids = [
    tokenizer(
        [x], add_special_tokens=False, return_attention_mask=False
        )['input_ids'][0] for x in  stop_tokens
    ]

stopping_criteria = StopOnTokens(stop_token_ids, tokenizer)
print(f"Stopping on tokens: {stopping_criteria.stop_token_ids}")

stopping_criteria = StoppingCriteriaList([stopping_criteria])

for batch_texts in tqdm(chunker(dataset['instruction'], batch_size)):
    
    batch_inputs = prepare_inputs_for_generation(batch_texts, tokenizer, 1000, truncate_from_start=False)

    input_ids = tokenizer(batch_inputs, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids['input_ids'].to(model.device), 
            max_new_tokens=100, 
            do_sample=True, 
            top_p=0.9, 
            temperature=0.9,
            stopping_criteria=stopping_criteria,
            )

    # strip off the prompt
    output_ids = output_ids[:, input_ids['input_ids'].size()[-1]:]
    
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for i, o in zip(batch_inputs, output_texts):
        print(i, '-->', o)
        print()
