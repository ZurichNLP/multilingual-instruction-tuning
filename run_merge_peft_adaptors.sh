#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# get list of paths from command line
in_dirs=("$@")

# loop over paths
for in_dir in "${in_dirs[@]}"; do
    echo "Merging PEFT adaptors in ${in_dir}"
    CUDA_VISIBLE_DEVICES=0 python merge_peft_adapter.py \
        --adapter_model_name_or_path "${in_dir}" \
        --output_dir "${in_dir}_merged"
done