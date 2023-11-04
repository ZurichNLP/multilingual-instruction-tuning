#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs the LLM evaluation harness for 1 or more given models
#  bash run_lm_eval_harness.sh 4 resources/models/llama_2_7b_hf_fr_merged resources/models/llama_2_7b_hf_es_merged resources/models/llama_2_7b_hf_ru_merged resources/models/llama_2_7b_hf_ca_merged

gpu=$1 # comma separated list of gpu ids or just one gpu id
models=("${@:2}")

for model in "${models[@]}"; do

    # get name of the model from the path 
    model_name=$(basename "$model")
    
    echo ""
    echo "Running evaluation harness for model: ${model_name}"
    echo ""
    
    CUDA_VISIBLE_DEVICES="${gpu}" python llm_eval_harness.py \
        --model="hf-causal" \
        --model_args="pretrained=${model},dtype=float16" \
        --device="cuda" \
        --batch_size 32 2>&1 | tee "data/lm_evals/${model_name}.log"
    
    echo ""
    echo "Finished evaluation harness for model: ${model_name}"
    echo ""

done

echo ""
echo "Finished running evaluation harness for all models: ${models[@]}"
echo ""
