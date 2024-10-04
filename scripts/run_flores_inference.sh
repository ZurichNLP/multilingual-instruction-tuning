#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs inference for a given set of models and test sets.
# Example call:
# bash run_flores_inference.sh \
# -d 0 \
# -m resources/models/llama_2_7b_hf_ml1_merged resources/models/llama_2_7b_hf_ml2_merged resources/models/llama_2_7b_hf_ml3_merged resources/models/llama_2_7b_hf_ml4_merged resources/models/llama_2_7b_hf_ml5_merged resources/models/llama_2_7b_hf_ml6_merged resources/models/llama_2_7b_hf_guanaco_merged \
# -t data/flores_devtest_en_zh.json

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

for model in "${models[@]}"; do
    for test_set in "${test_sets[@]}"; do

        # get target lang tag from test set data/flores_devtest_en_fr.json -> fr
        target_lang=$(echo "${test_set}" | awk -F'_' '{print $NF}' | awk -F'.' '{print $1}')
        echo "Language tag: ${target_lang}"

        # get different languages for prompts, expects en and target language
        for prompt_lang in "en" "${target_lang}"; do
            echo "Prompt language: ${prompt_lang}"

            for seed in "${seeds[@]}"; do
                echo ""
                echo "${model} --- ${test_set} --- ${seed} --- ${d} --- ${target_lang} --- ${prompt_lang}"
                echo ""

                CUDA_VISIBLE_DEVICES="${d}" python -m inference "${model}" \
                    --input_file "${test_set}" \
                    --batch_size 128 \
                    --temperature 0.9 \
                    --seed "${seed}" \
                    --output_path resources/outputs/flores \
                    --prompt_format "prompts/flores_en_${target_lang}_${prompt_lang}" \
                    --src_key source --tgt_key reference \
                    --stop "### Human:" "### Assistant:" "### Human" "### Assistant" \
                    --n_gpus "${n_gpus}"
            done
        done
    done
done
