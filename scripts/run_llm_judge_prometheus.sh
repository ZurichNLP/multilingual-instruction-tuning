#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs the evaluation pipeline for the LLM judge.
# It takes as input the outputs of the LLM models and outputs the evaluation results.

# ACHTUNG: Run conda activate prometheus before running this script!!!

# Example usage:
# direct vs translated evaluation ablation:
# nohup bash scripts/run_llm_judge_prometheus.sh -m llama_2_7b_hf_ml2_merged -l de en fr zh ru es ca sv bg no is hi el hu fi vi ko -gpu 0 -limit 300 -t true > logs/prometheus_direct_vs_translate_eval.log &
# nohup bash scripts/run_llm_judge_prometheus.sh -m llama_2_7b_hf_ml1_merged llama_2_7b_hf_ml3_merged llama_2_7b_hf_ml4_merged llama_2_7b_hf_ml2_merged llama_2_7b_hf_ml5_merged llama_2_7b_hf_ml6_merged llama_2_7b_hf_guanaco_merged -l de en fr zh ru es ca sv bg no is hi el hu fi vi ko -gpu 3 -limit 300 > logs/prometheus_direct_eval.log &
# bash scripts/run_llm_judge_prometheus.sh -m llama_2_7b_hf_ml2_merged -l de fr zh ru -gpu 0 -limit 300
# bash scripts/run_llm_judge_prometheus.sh -m llama_3_8b_ml2_merged -l de fr zh ru -gpu 0 -limit 300 -t

set -e

# Variables to hold arguments for -m and -t
declare -a models
declare -a langs
seeds=(0 42 723) # default seeds
translation_model="gpt-3.5-turbo-1106"
evaluation_model="prometheus-eval/prometheus-7b-v2.0"
evaluate_with_translations=false
gpu="0"
limit="-1"

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
        -gpu)
            shift
            gpu="$1"
            shift
            ;;
        -limit)
            shift
            limit="$1"
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

        # find all infile paths
        infiles=$( (find "data/alpaca_eval_outputs/${model}" -type f -name "alpaca_eval_instructions_${lang}-none-guanaco_prompt-s*.jsonl") )
        
        for infile in ${infiles[@]}; do
    
            echo ""
            echo "*** Running evaluation pipeline for model: ${infile} ***"
            echo ""
    
            if [ "$evaluate_with_translations" = true ] && [ "${lang}" != "en" ]; then
    
                echo "*** Translating non-English responses to English... ***"
                
                translated_infile="data/alpaca_eval_outputs_translated/${model}/$(basename "$infile")"
                
                # step 1: translate non-English responses to English
                python translate_with_gpt.py \
                    --input_file "${infile}" \
                    --output_file "${translated_infile}" \
                    --tgt_lang "English" \
                    --src_key "system" \
                    --limit "${limit}" --data_seed 42 --api_seed 42 \
                    --model_name "${translation_model}" \
                    --dataset_type "alpaca_eval_outputs" --original_prompts "data/alpaca_eval/alpaca_eval_instructions_en.json"

                echo "*** Evaluating translated responses... ***"
                # step 2: evaluate translated English responses
                CUDA_VISIBLE_DEVICES="${gpu}" python llm_judge_prometheus.py \
                    --input_file "${translated_infile}" \
                    --eval_model_name "${evaluation_model}" \
                    --src_key "source_en" \
                    --tgt_key "system_en" \
                    --data_seed 42

            fi               

            echo "*** Evaluating non-translated responses directly... ***"
            # step 1: evaluate the original English responses
            CUDA_VISIBLE_DEVICES="${gpu}" python llm_judge_prometheus.py \
                --input_file "${infile}" \
                --eval_model_name "${evaluation_model}" \
                --eval_model_name "${evaluation_model}" \
                --src_key "source" \
                --tgt_key "system" \
                --limit "${limit}" --data_seed 42


        done
    done
done
