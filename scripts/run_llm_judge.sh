#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs the evaluation pipeline for the LLM judge.
# It takes as input the outputs of the LLM models and outputs the evaluation results.

# Example usage:
# bash run_llm_judge.sh -m llama_2_7b_hf_ml2_merged -l de fr


set -e

# Variables to hold arguments for -m and -t
declare -a models
declare -a langs
seeds=(0 42 723) # default seeds
translation_model="gpt-3.5-turbo-1106"
evaluation_model="gpt-3.5-turbo-1106"
evaluate_with_translations=false

# Loop to parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -m)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                models+=("$1")
                shift
            done
            ;;
        -l)
            shift
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                langs+=("$1")
                shift
            done
            ;;
        -s)
            shift
            seeds=() # reset seeds and use the ones provided
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                seeds+=("$1")
                shift
            done
            ;;
        -t)
            shift
            evaluate_with_translations=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# check if variables are set
if [ -z "${models}" ]; then
    echo "Please set -m"
    exit 1
fi

if [ -z "${langs}" ]; then
    echo "Please set -l"
    exit 1
fi

echo "models: ${models[@]}"
echo "langs: ${langs[@]}"
echo "seeds: ${seeds[@]}"
echo "evaluate_with_translations: ${evaluate_with_translations}"

for model in "${models[@]}"; do
    for lang in "${langs[@]}"; do
        for seed in "${seeds[@]}"; do
    
            echo ""
            echo "*** Running evaluation pipeline for model: ${model}, lang: ${lang}, seed: ${seed} ***"
            echo ""

            if [ "$evaluate_with_translations" = true ] && [ "${lang}" != "en" ]; then
    
                echo "*** Translating non-English responses to English... ***"
                # step 1: translate non-English responses to English
                python translate_with_gpt.py \
                    --input_file "data/alpaca_eval_outputs/${model}/alpaca_eval_instructions_${lang}-none-guanaco_prompt-s${seed}-k50-p0.9-t0.8-b8.jsonl" \
                    --output_file "data/alpaca_eval_outputs_translated/${model}/alpaca_eval_instructions_${lang}-none-guanaco_prompt-s${seed}-k50-p0.9-t0.8-b8.jsonl" \
                    --tgt_lang "English" \
                    --src_key "system" \
                    --limit 300 --data_seed 42 --api_seed 42 \
                    --model_name "${translation_model}" \
                    --original_prompts "data/alpaca_eval_instructions_en.json"

                echo "*** Evaluating translated responses... ***"
                # step 2: evaluate translated English responses
                python llm_judge.py \
                    --input_file "data/alpaca_eval_outputs_translated/${model}/alpaca_eval_instructions_${lang}-none-guanaco_prompt-s${seed}-k50-p0.9-t0.8-b8.jsonl" \
                    --eval_model_name "${evaluation_model}" \
                    --src_key "source_en" \
                    --tgt_key "system_en" \
                    --api_seed 42 --data_seed 42

            fi               

            echo "*** Evaluating non-translated responses directly... ***"
            # step 1: evaluate the original English responses
            python llm_judge.py \
                --input_file "data/alpaca_eval_outputs/${model}/alpaca_eval_instructions_${lang}-none-guanaco_prompt-s${seed}-k50-p0.9-t0.8-b"*".jsonl" \
                --eval_model_name "${evaluation_model}" \
                --src_key "source" \
                --tgt_key "system" \
                --limit 300 --api_seed 42 --data_seed 42 \
                --api_seed 42 --data_seed 42


        done
    done
done


