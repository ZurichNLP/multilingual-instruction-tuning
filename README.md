This repository contains the code and data for the paper "Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?"

# Environment Setup

To setup the environment, we recommend using conda, e.g.:

```bash
conda create -n ml_llm -c conda-forge python=3.10 cudatoolkit=11.8 -y
conda activate ml_llm
pip install vllm==0.2.1
pip install -r requirements.txt
```

Download model used for language detection to `resources/lid/`

```bash
mkdir resources
wget https://data.statmt.org/lid/lid201-model.bin.gz -P resources/lid/
gzip -d resources/lid/lid201-model.bin.gz 
```

For evaluations using [Eleuther AI's LM Evaluation Harness](<https://github.com/EleutherAI/lm-evaluation-harness>), run:

```bash
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
git reset --hard 3ccea2b2
pip install -e ".[multilingual]"
```

# API Keys

If running experiments with OpenAI's API-based models, create a file containing your API key, e.g.:

```bash
echo "OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'" > api_secrets.py
```

# Models

All models training datasets used in our experiments are available on the [Hugging Face Hub](https://huggingface.co/collections/tannonk/multilingual-instruction-tuning-65855e8d92eba5ad69df4b2a).

# Data

The data used for our experiments is available in [data/](./data).

This includes:
    - Guanaco and its subsets (Mono, Multi-2, Multi-3, etc.)
    - [Alpaca Eval prompts](.data/alpaca_eval) in different languages (used for single-turn dialogue evaluation)
    - [MultiSim](./data/multisim) simplification benchmark (used for sentence simplification evaluation)
    - [XQuAD](./data/xquad) (used for extractive QA evaluation)
    - [X-CSQA](./data/xcsqa) (used for commonsense reasoning evaluation)

Where applicable, we include the prompt templates used to run the evaluations with each dataset.

For reproducibility, the data can be prepared from the original sources using the relevant notebooks in [data_prep/](./data_prep).

## Model Training

To train a model on a given dataset, use the script `sft_training.py`. For example:

```bash
CUDA_VISIBLE_DEVICES=2,3 nohup python sft_training.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --train_dataset "data/guanaco/guanaco_train_ml2.json" \
    --eval_dataset "data/guanaco/guanaco_test.json" \
    --output_dir "resources/models/llama_2_7b_hf_ml2" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 \
    --log_with "wandb" >| resources/models/logs/llama_2_7b_hf_ml2.log &
```

Once training is completed, we merge the learned adapters with the base model for easy loading with [vLLM](https://github.com/vllm-project/vllm).

```bash
python merge_peft_adapter.py \
    --adapter_model_name_or_path "resources/models/llama_2_7b_hf_ml2" \
    --output_dir "resources/models/llama_2_7b_hf_ml2_merged"
```

# Inference

To run inference for the different tasks, you can use the appropriate `run_inference*.sh` script ([here](./scripts)), specifying the GPU device ID, model directories and evaluation datasets.

### Single-turn Dialogue

```bash
bash scripts/run_alpaca_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/alpaca_eval/alpaca_eval_instructions_is.json data/alpaca_eval/alpaca_eval_instructions_el.json data/alpaca_eval/alpaca_eval_instructions_hi.json
```

### Sentence Simplification

```bash
bash scripts/run_ts_inference.sh -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/multisim/en-en.json data/multisim/en-de.json data/multisim/de-de.json 
```

### X-CSQA

```bash
bash scripts/run_xcsqa_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/xcsqa/xcsqa_dev_zh_zh.json data/xcsqa/xcsqa_dev_fr_fr.json
```

### XQuAD

```bash
bash scripts/run_xnli_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/xquad/xquad_dev_en_hi.json data/xquad/xquad_dev_hi_hi.json
```

### XNLI

```bash
nohup bash scripts/run_lm_eval_harness.sh 0 resources/models/llama_2_7b_hf_ml2_merged >| logs/llama_2_7b_hf_ml2_merged.log &
```

# Evaluation

The script [run_llm_judge.sh](./scripts/run_llm_judge.sh), can be used to evaluate chat responses given multiple models and target languages.
E.g.:

```bash
bash scripts/run_llm_judge.sh \
    -m data/alpaca_eval_outputs/llama_2_7b_hf_ml2_merged data/alpaca_eval_outputs/llama_2_7b_hf_ml3_merged \
    -l is el hi
```

# Results

Plots from the paper can be generated using [this notebook](./process_main_results.ipynb).
This assumes the model outputs and evaluation results are available in the following directory: `./resources/outputs`.

# Citation

```
@misc{kew2023turning,
      title={Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?}, 
      author={Tannon Kew and Florian Schottmann and Rico Sennrich},
      year={2023},
      eprint={2312.12683},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
