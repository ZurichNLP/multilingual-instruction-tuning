#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference for a given set of models and test sets.
# Example call:
# bash scripts/run_ts_inference.sh -d 7 -m resources/models/llama_2_7b_hf_ml1_merged -t data/multisim/de-de.json

# Variables to hold arguments for -m and -t
declare -a models
declare -a test_sets
seeds=(0 42 723)

# Loop to parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -d)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                d=("$1")
                shift
            done
            ;;
        -m)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                models+=("$1")
                shift
            done
            ;;
        -t)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                test_sets+=("$1")
                shift
            done
            ;;
        -s)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                seeds+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# check if variables are set
if [ -z "${d}" ]; then
    echo "Please set -d"
    exit 1
fi

if [ -z "${models}" ]; then
    echo "Please set -m"
    exit 1
fi

if [ -z "${test_sets}" ]; then
    echo "Please set -t"
    exit 1
fi

n_gpus=$(echo "$d" | awk -F',' '{print NF}')

echo "Device ids: ${d}"
echo "Number of GPUs: ${n_gpus}"
echo "Seeds: ${seeds[@]}"
echo "Models: ${models[@]}"
echo "Test sets: ${test_sets[@]}"

mkdir -p resources/outputs/multisim

for model in "${models[@]}"; do
    for test_set in "${test_sets[@]}"; do
        for seed in "${seeds[@]}"; do
            echo ""
            echo "${model} --- ${test_set} --- ${seed} --- ${d}"
            echo ""

            CUDA_VISIBLE_DEVICES="${d}" python -m inference "${model}" \
                --input_file "${test_set}" \
                --batch_size 128 \
                --seed "${seed}" \
                --output_path "resources/outputs/multisim" \
                --prompt_format "prompts/blank" \
                --src_key "instruction" --tgt_key "target" --orig_src_key "source" \
                --stop "### Human:" "### Assistant:" "### Human" "### Assistant" \
                --n_gpus "${n_gpus}" \
                --force True


        done
    done
done
