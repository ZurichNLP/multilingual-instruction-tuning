#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script extends the script run_evaluations.sh for additional tasks rrgen and ts
# Example usage:
# bash scripts/run_evaluation_with_sari.sh 0 resources/outputs/rrgen/llama_2_7b_hf_ml*
# bash scripts/run_evaluation_with_sari.sh 0 resources/outputs/multisim/llama_2_7b_hf_ml1_merged/

set -e

gpu=$1 # comma separated list of gpu ids or just one gpu id
model_outputs_dirs=("${@:2}")

for model_outputs_dir in "${model_outputs_dirs[@]}"; do
    for model_outputs_file in "${model_outputs_dir}"/*".jsonl"; do

        output_file="${model_outputs_file%.*}.eval"

        file_name=$(basename "${model_outputs_file}")
        # extract target language from filename (e.g. en_de-none-blank-s42-k50-p0.9-t0.8-b128.jsonl -> de)
        lang=$(echo "${model_outputs_file}" | grep -oP "_\w\w-" | sed 's/_//g' | sed 's/-//g')
        
        echo "Running evaluation on ${model_outputs_file}"
        echo "Inferred language: ${lang}"
        echo "Output file: ${output_file}"        
        
        # run lang assignment
        python assign_langs.py "${model_outputs_file}" --keys "orig_source" "system"
        
        CUDA_VISIBLE_DEVICES="${gpu}" python evaluation_with_sari.py \
            "${model_outputs_file}" \
            --lang "${lang}" \
            --output_file "${output_file}" \
            --src_key "orig_source" --ref_key "reference" --tgt_key "system" \
            --force

    done
done
