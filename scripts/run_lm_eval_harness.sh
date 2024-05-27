#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script runs the LLM evaluation harness for 1 or more given models
# bash scripts/run_lm_eval_harness.sh 0 resources/models/llama_2_7b_hf_ml1_merged resources/models/llama_2_7b_hf_ml2_merged 
# bash scripts/run_lm_eval_harness.sh 1 resources/models/llama_2_7b_hf_ml3_merged resources/models/llama_2_7b_hf_ml4_merged 
# bash scripts/run_lm_eval_harness.sh 2 resources/models/llama_2_7b_hf_ml5_merged resources/models/llama_2_7b_hf_ml6_merged 
# bash scripts/run_lm_eval_harness.sh 7 resources/models/llama_2_7b_hf_guanaco_merged meta-llama/Llama-2-7b-hf

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
        --output_base_path "data/lm_eval_harness" \
        --model_args="pretrained=${model},dtype=float16" \
        --device="cuda" \
        --batch_size 32 2>&1 | tee "logs/lm_evals_harness-${model_name}.log"
    
    echo ""
    echo "Finished evaluation harness for model: ${model_name}"
    echo ""

done

echo ""
echo "Finished running evaluation harness for all models: ${models[@]}"
echo ""
