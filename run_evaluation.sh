#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference on the test set using the trained model.
# Example usage:
# bash run_evaluation.sh 0 data/outputs/llama_2_7b_hf_ml*
# bash run_evaluation.sh 0 data/outputs/llama_2_7b_hf_zh_merged

gpu=$1 # comma separated list of gpu ids or just one gpu id
model_outputs_dirs=("${@:2}")

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        output_file="${model_outputs_file%.*}.eval"
        # expected filename: /../../alpaca_eval_instructions_de-none-guanaco_prompt-s42-k50-p0.9-t0.8-b8.jsonl
        lang=$(echo "${model_outputs_file}" | grep -oP "(?<=alpaca_eval_instructions_)[a-z]{2}")

        if [ -f "${output_file}" ]; then
            echo "Output file ${output_file} already exists, skipping..."
        else
            # run lang assignment
            python assign_langs.py "${model_outputs_file}"
            echo "Running evaluation on ${model_outputs_file}"
            echo "Inferred language: ${lang}"
            echo "Output file: ${output_file}"        
            CUDA_VISIBLE_DEVICES="${gpu}" python evaluation.py \
                "${model_outputs_file}" \
                --lang "${lang}" \
                --output_file "${output_file}" #--use_cuda
        fi

    done
done
