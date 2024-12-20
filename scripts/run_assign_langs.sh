#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script computes the evaluations for a set model outputs.
# Example usage:
# bash scripts/run_assign_langs.sh resources/outputs/alpaca_eval/llama_2_7b_hf_ml*

model_outputs_dirs=("${@:1}")

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        output_file="${model_outputs_file%.*}.eval"
        # expected filename: /../../alpaca_eval_instructions_de-none-guanaco_prompt-s42-k50-p0.9-t0.8-b8.jsonl
        lang=$(echo "${model_outputs_file}" | grep -oP "(?<=alpaca_eval_instructions_)[a-z]{2}")
        
        # run lang assignment
        echo "Running language id on ${model_outputs_file}. Inferred lang: ${lang}"
        python assign_langs.py "${model_outputs_file}"
        
    done
done
