#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# get list of paths from command line

gpu=$1 # comma separated list of gpu ids or just one gpu id
in_dirs=("${@:2}")

# loop over paths
for in_dir in "${in_dirs[@]}"; do
    echo "Merging PEFT adaptors in ${in_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" python merge_peft_adapter.py \
        --adapter_model_name_or_path "${in_dir}" \
        --output_dir "${in_dir}_merged"
done