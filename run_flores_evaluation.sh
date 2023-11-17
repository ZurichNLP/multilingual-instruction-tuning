#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference on the test set using the trained model.
# Example usage:
# bash run_flores_evaluation.sh 0 data/outputs/llama_2_7b_hf_ml*
# CUDA_VISIBLE_DEVICES=7 python -m evaluation data/flores_outputs/llama_2_7b_hf_ml1_merged/flores_devtest_en_de-none-flores_en_de_en-s0-k50-p0.9-t0.001-b128.jsonl --lang de --src_key source --tgt_key system --ref_key reference

gpu=$1 # comma separated list of gpu ids or just one gpu id
model_outputs_dirs=("${@:2}")

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        output_file="${model_outputs_file%.*}.eval"
        # expected filename: /../../flores_devtest_en_de-none-flores_en_de_en-s0-k50-p0.9-t0.001-b128.jsonl
        lang=$(basename "${model_outputs_file}" | awk -F'[_-]' '{print $4}')

        # run lang assignment
        echo "Running evaluation on ${model_outputs_file}"
        echo "Inferred language: ${lang}"
        echo "Output file: ${output_file}"        
        CUDA_VISIBLE_DEVICES="${gpu}" python evaluation.py \
            "${model_outputs_file}" \
            --lang "${lang}" \
            --output_file "${output_file}" \
            --src_key source --tgt_key system --ref_key reference --force

    done
done
