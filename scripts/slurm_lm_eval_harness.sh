#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=512GB
#SBATCH --time=23:59:00
#SBATCH --partition=lowprio

# This script submits a slurm jobs to run the LLM evaluation harness for 1 or more given models
# sbatch slurm_lm_eval_harness.sh resources/models/llama_2_70b_hf_mt_ml1_merged resources/models/llama_2_70b_hf_mt_ml2_merged resources/models/llama_2_70b_hf_mt_ml3_merged 

# hardcoded defaults
BASE="/data/tkew/projects/multilingual-instruction-tuning" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

module purge
module load anaconda3 multigpu a100

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate vllm && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "${BASE}" && echo $(pwd) || exit 1


models=("${@:1}")

echo "models: ${models[@]}"

for model in "${models[@]}"; do

    # get name of the model from the path 
    model_name=$(basename "$model")
    
    echo ""
    echo "Running evaluation harness for model: ${model_name}"
    echo ""
    
    python llm_eval_harness.py \
        --model="hf-causal-experimental" \
        --model_args="pretrained=${model},load_in_8bit=True" \
        --device="cuda" \
        --batch_size 16 2>&1 | tee "data/lm_evals/${model_name}.log"

    # if previous command failed, exit
    if [ $? -ne 0 ]; then
        echo "Failed to run evaluation harness for model: ${model_name}"
        exit 1
    fi
        echo ""
        echo "Finished evaluation harness for model: ${model_name}"
        echo ""
done

echo ""
echo "Finished running evaluation harness for all models: ${models[@]}"
echo ""
