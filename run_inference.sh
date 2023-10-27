#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference on the test set using the trained model.

gpu=$1 # comma separated list of gpu ids or just one gpu id
model_path=$2
# all remaining arguments are list of datasets to run inference on
datasets=("${@:3}")
seeds=(0 42 723)

n_gpus=$(echo "$gpu" | awk -F',' '{print NF}')

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo ""
        echo "${model_path} --- ${dataset} --- ${seed} --- ${gpu}"
        echo ""
        CUDA_VISIBLE_DEVICES="${gpu}" python -m inference "${model_path}" \
            --input_file "${dataset}" \
            --batch_size 8 \
            --seed "${seed}" \
            --output_path data/outputs \
            --prompt_format guanaco_prompt \
            --src_key instruction \
            --stop "### Human:" "### Assistant:" "### Human" "### Assistant" \
            --n_gpus "${n_gpus}"
    done
done

