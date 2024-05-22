#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script computes the evaluations for a set model outputs.
# Example usage:
# bash scripts/run_evaluation.sh 0 data/alpaca_eval_outputs/llama_2_7b_hf_ml*

gpu=$1 # comma separated list of gpu ids or just one gpu id
model_outputs_dirs=("${@:2}")

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        output_file="${model_outputs_file%.*}.eval"
        # expected filename: /../../alpaca_eval_instructions_de-none-guanaco_prompt-s42-k50-p0.9-t0.8-b8.jsonl
        lang=$(echo "${model_outputs_file}" | grep -oP "(?<=alpaca_eval_instructions_)[a-z]{2}")
        
        # run lang assignment
        python assign_langs.py "${model_outputs_file}"
        echo "Running evaluation on ${model_outputs_file}"
        echo "Inferred language: ${lang}"
        echo "Output file: ${output_file}"        
        CUDA_VISIBLE_DEVICES="${gpu}" python evaluation.py \
            "${model_outputs_file}" \
            --lang "${lang}" \
            --output_file "${output_file}" --force

    done
done
