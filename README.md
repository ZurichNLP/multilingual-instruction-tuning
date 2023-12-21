This repository contains the code and data for the paper "Turning English-centric LLMs into Polyglots: How Much Multilinguality is Needed?"

# Environment Setup

To setup the environment, we recommend using conda, e.g.:

```
conda create -n ml_llm -c conda-forge python=3.10 cudatoolkit=11.8 -y
conda activate ml_llm
pip install vllm
pip install -r requirements.txt
```

Download model used for language detection to `resources/lid/`

```
mkdir resources
wget https://data.statmt.org/lid/lid201-model.bin.gz -P resources/lid/
gzip -d resources/lid/lid201-model.bin.gz 
```

For evaluations using [Eleuther AI's LM Evaluation Harness](<https://github.com/EleutherAI/lm-evaluation-harness>), run:

```
git clone git@github.com:EleutherAI/lm-evaluation-harness.git
git reset --hard 3ccea2b2
pip install -e ".[multilingual]"
```

# API Keys

If running experiments with OpenAI's API-based models, create a file containing your API key, e.g.:

```
echo "OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'" > api_secrets.py
```

# Data and Data Preparation

The data used for our experiments is available in [data](./data) and can be prepared from the original sources using the scripts [here](./data_prep).

# Model Training

To train a model on a given dataset, use the script `sft_training.py`. For example:

```
CUDA_VISIBLE_DEVICES=2,3 nohup python sft_training.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --train_dataset "data/guanaco/guanaco_train_ml2.json" \
    --eval_dataset "data/guanaco/guanaco_test.json" \
    --output_dir "resources/models/llama_2_7b_hf_ml2" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 \
    --log_with "wandb" >| resources/models/logs/llama_2_7b_hf_ml2.log &
```

Once training is completed, we merge the learned adapters with the base model for easy loading with vLLM.

```
python merge_peft_adapter.py \
    --adapter_model_name_or_path "resources/models/llama_2_7b_hf_ml2" \
    --output_dir "resources/models/llama_2_7b_hf_ml2_merged"
```

# Inference

To run inference for the different tasks, use the appropriate `run_inference*.sh` script, specifying the GPU device ID, model directories and evaluation datasets.

### Chat

```
bash scripts/run_alpaca_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/alpaca_eval/alpaca_eval_instructions_is.json data/alpaca_eval/alpaca_eval_instructions_el.json data/alpaca_eval/alpaca_eval_instructions_hi.json
```

### X-CSQA

```
bash scripts/run_xcsqa_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/xcsqa/xcsqa_dev_zh_zh.json data/xcsqa/xcsqa_dev_fr_fr.json
```

### XQuAD

```
bash scripts/run_xnli_inference.sh \
    -d 0 \
    -m resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged \
    -t data/xquad/xquad_dev_en_hi.json data/xquad/xquad_dev_hi_hi.json
```

### XNLI

```
nohup bash scripts/run_lm_eval_harness.sh 0 resources/models/llama_2_7b_hf_ml2_merged >| logs/llama_2_7b_hf_ml2_merged.log &
```

# Evaluation

The script [run_llm_judge.sh](./scripts/run_llm_judge.sh), can be used to evaluate chat responses given multiple models and target languages.
E.g.:

```
bash scripts/run_llm_judge.sh \
    -m data/alpaca_eval_outputs/llama_2_7b_hf_ml2_merged data/alpaca_eval_outputs/llama_2_7b_hf_ml3_merged \
    -l is el hi
```

# Results

Plots used in the paper can be generated using [this notebook](./process_main_results.ipynb).

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
