from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

"""
Example call:
    python merge_peft_adapter.py \
        --adapter_model_name_or_path <path_to_finetuned_model \
        --output_dir <output_path>"
"""

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "name or path of the Adapter to be merged"})
    base_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "name or path to base model"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "path to merged output model"})
    compute_dtype: Optional[str] = field(default="fp16", metadata={"help": "compute dtype of the model"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
assert args.adapter_model_name_or_path is not None, "please provide the name of the Adapter you would like to merge"

compute_dtype = (
    torch.float16 if args.compute_dtype == 'fp16' else (
        torch.bfloat16 if args.compute_dtype == 'bf16' else torch.float32
        )
    )
print(f"Using {compute_dtype} for loading")

peft_config = PeftConfig.from_pretrained(args.adapter_model_name_or_path)

if args.base_model_name_or_path is None:
    base_model_name_or_path = peft_config.base_model_name_or_path
    print(f"Inferred base model name or path from Adapter config: {base_model_name_or_path}")
else:
    base_model_name_or_path = args.base_model_name_or_path

print(f"Loading base model from {base_model_name_or_path}")
if peft_config.task_type == "SEQ_CLS":
    # peft is for reward model so load sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path, num_labels=1, torch_dtype=compute_dtype, trust_remote_code=False
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, return_dict=True, torch_dtype=compute_dtype, trust_remote_code=False
    )

print(f"Loading tokenizer from {base_model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

print(f"Loading Adapter from {args.adapter_model_name_or_path}")
model = PeftModel.from_pretrained(model, args.adapter_model_name_or_path)
model.eval()

model = model.merge_and_unload()

print(f"Saving model with merged adaptor weights and tokenizer to {args.output_dir}")
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

model.save_pretrained(f"{args.output_dir}")
tokenizer.save_pretrained(f"{args.output_dir}")

print("Done!")