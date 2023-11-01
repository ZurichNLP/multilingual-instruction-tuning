#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference on the test set using the trained model.
# Example usage:
# bash run_llm_evaluation.sh 0 data/outputs/llama_2_7b_hf_ml*
# bash run_llm_evaluation.sh data/outputs/llama_2_7b_hf_zh_merged

model_outputs_dirs=("${@:1}")
eval_model="gpt-3.5-turbo"
# seed=42

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    # for model_outputs_file in "${model_outputs_dir}"/*"s${seed}"*".jsonl"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        echo "Running LLM-judge evaluation on ${model_outputs_file}"
        
        python llm_judge.py \
            --input_file "${model_outputs_file}" \
            --model_name "${eval_model}" \
            --seed 42 \
            --limit 50 \
            --src_key "source" --tgt_key "system"
    
    done
done
