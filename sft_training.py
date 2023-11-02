import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig
)

import bitsandbytes as bnb

from trl import SFTTrainer
from trl.trainer import (
    ConstantLengthDataset, 
)



@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."},
    )
    
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum sequence length that this model might ever be used with. Typically 512, 1024, 2048."},
    )

    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "The maximum number of steps to train for."},
    )
    
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "The number of epochs to train for."},
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "The batch size per GPU for training."},
    )

    per_device_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "The batch size per GPU for evaluation."},
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=2,
        metadata={"help": "The number of gradient accumulation steps."},
    )
    
    evaluation_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use. One of ['no', 'steps', 'epoch']."},
    )

    save_total_limit: Optional[int] = field(
        default=10,
        metadata={"help": "The maximum number of checkpoints to save."},
    )

    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Learning rate."},
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The learning rate scheduler to use. One of ['constant', 'cosine', 'cosine_with_restarts', 'polynomial', 'linear', 'linear_with_warmup']."},
    )

    warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of steps for the warmup. Note overrides warmup_ratio."},
    )

    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "Fraction of steps to do a warmup for."},
    )

    weight_decay: Optional[float] = field(
        default=0.05,
        metadata={"help": "Weight decay."},
    )

    local_rank: Optional[int] = field(
        default=-1,
        metadata={"help": "Local rank."},
    )

    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use fp16 precision instead of 32-bit"}
    )

    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use bf16-bit (mixed) precision instead of 32-bit"}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing to save memory at the expense of slower backward pass."}
    )

    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Random seed."},
    )

    num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of workers."},
    )

    output_dir: Optional[str] = field(
        default="./checkpoints",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "The frequency at which logs are printed."},
    )

    wandb_project: Optional[str] = field(
        default="mllm",
        metadata={"help": "The name of the project for wandb logging."},
    )

    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the run for wandb logging."},
    )

    eval_steps: Optional[int] = field(
        default=200,
        metadata={"help": "The frequency at which evaluation is performed."},
    )

    save_steps: Optional[int] = field(
        default=200,
        metadata={"help": "The frequency at which checkpoints are saved."},
    )
    
    log_with: Optional[str] = field(
        default="none",
        metadata={"help": "The logging backend to use. One of ['wandb', 'tensorboard', 'none']."},
    )

    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset to use for training."},
    )

    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset to use for evaluation."},
    )

    lang: Optional[str] = field(
        default=None,
        metadata={"help": "The language to use."},
    )

    max_train_instances: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of training instances to use."},
    )

    max_eval_instances: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of evaluation instances to use."},
    )

    optim: Optional[str] = field(
        default='paged_adamw_32bit', 
        metadata={"help": 'The optimizer to be used'}
    )

    max_grad_norm: Optional[float] = field(
        default=0.3,
        metadata={"help": "The maximum gradient norm. https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L205C5-L205C18"},
    )

    lora_r: Optional[int] = field(
        default=64,
        metadata={"help": "The rank of the update matrices, expressed in int. "
                  "Lower rank results in smaller update matrices with fewer trainable parameters. "
                  "If set to 0, the update matrices are not used (i.e. the model is a standard Transformer)."},
    )
    
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "The scaling factor for the update matrices. "}
    )

    lora_bias: Optional[str] = field(
        default='none',
        metadata={"help": "Specifies if the bias parameters should be trained. Can be 'none', 'all' or 'lora_only'."}
    )
    
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "Specifies the dropout rate for the LoRA layers."}
    )
    
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Trust remote code for Falcon and MPT models."},
    )

    bits: Optional[int] = field(
        default=16,
        metadata={"help": "4 or 8bit precision base model loading."},
    )

    pack_sequences: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to pack sequences into a ConstantLengthDataset or not."},
    )


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    avg_tokens_per_example = total_tokens / nb_examples
    avg_chars_per_token = total_characters / total_tokens 
    return avg_chars_per_token, avg_tokens_per_example


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    if example.get('text'):
        text = example['text']
    else:
        text = f"{example['input']} {example['label']}"
    return text

def format_prompts(examples):
    """"""
    processed_data = []

    try:
        for i, l in zip(examples.format('input'), examples.format('label')):
            processed_data.append(f"{i}{l}")
    except:
        for i in examples.format('text'):
            processed_data.append(i)
    
    print(f"processed_data[0]: {processed_data[0]}")
    return processed_data


def find_all_linear_names(args, model):
    """From qlora.py"""
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main(args):
    
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_with == "wandb":
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, group="sft", job_type="sft")

    print(f"Arguments: {args}")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load dataset
    print(f"Loading the following datasets: {args.train_dataset}, {args.eval_dataset}")    
    train_dataset = load_dataset("json", data_files=args.train_dataset)
    eval_dataset = load_dataset("json", data_files=args.eval_dataset)

    def get_tokenized_length(example):
        example['length'] = len(tokenizer(example['text']).tokens())
        return example

    # add length to dataset
    train_dataset = train_dataset.map(get_tokenized_length, batched=False, batch_size=1)
    eval_dataset = eval_dataset.map(get_tokenized_length, batched=False, batch_size=1)

    # sort dataset by length
    train_dataset = train_dataset.sort("length")
    eval_dataset = eval_dataset.sort("length")

    train_dataset = train_dataset['train']
    eval_dataset = eval_dataset['train']

    print(f"train dataset sample:")
    print(f"train_dataset [0]: {train_dataset['text'][0]}")
    print(f"train_dataset [-1]: {train_dataset['text'][-1]}")
    
    # model
    print("Loading the model")
    
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    print(f"Compute dtype: {compute_dtype}")

    if args.bits in [4, 8]:
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True if compute_dtype == torch.float16 else False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        use_cache=False,
    )
    
    # update config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if args.lora_r > 0:

        # freeze model weights and enable gradient checkpointing
        # https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L376
        # Note: if not using 4 or 8 bit, we need to manually enable gradient checkpointing
        if args.bits not in [4, 8]:
            model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    
        print(f'adding LoRA modules...')
        
        # automatically find target modules (all linear layers) 
        # (as done in https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L385C19-L385C19
        target_modules = find_all_linear_names(args, model)

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha, # 64
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    
        print(f"LoRA config: {peft_config}")
        model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)
    
    train_dataset.start_iteration = 0
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy=args.evaluation_strategy if eval_dataset is not None else "no",
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_total_limit=args.save_total_limit,
        fp16=True, # avoid converting inputs
        # fp16=compute_dtype == torch.float16,
        # bf16=compute_dtype == torch.bfloat16,
        weight_decay=args.weight_decay,
        report_to=args.log_with,
        ddp_find_unused_parameters=False, # avoid RuntimeError: Expected to mark a variable ready only once. (https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/supervised_finetuning.py)
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        local_rank=args.local_rank,
    )

    if args.pack_sequences:
        # pack sequences into a ConstantLengthDataset
        chars_per_token, tokens_per_example = chars_token_ratio(train_dataset, tokenizer)

        print(f"**** Dataset statistics ****")
        print(f"chars_per_token: {chars_per_token}")
        print(f"tokens_per_example: {tokens_per_example}")

        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_dataset,
            formatting_func=prepare_sample_text,
            infinite=True,
            seq_length=args.max_seq_length,
            chars_per_token=chars_per_token,
        )

        if eval_dataset is not None:
            eval_dataset = ConstantLengthDataset(
                tokenizer,
                eval_dataset,
                formatting_func=prepare_sample_text,
                infinite=False,
                seq_length=args.max_seq_length,
                chars_per_token=chars_per_token, 
            )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=args.max_seq_length,
        formatting_func=format_prompts, # used if not packing sequences
        # infinite=True, # avoid early stopping due to packing https://github.com/lvwerra/trl/issues/450
    )

    
    print_trainable_parameters(trainer.model)

    print("Training...")
    
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(args.output_dir)
    print(f"Model saved in {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer saved in {args.output_dir}")

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)