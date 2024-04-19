#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=256GB
#SBATCH --time=23:59:00
#SBATCH --partition=lowprio

# Usage: sbatch slurm_xquad_inference.sh -m <model_name_or_path> -t <test_datasets>
# sbatch slurm_xquad_inference.sh -m resources/models/llama_2_70b_hf_mt_ml1_merged -t data/xquad_dev_*

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

# Variables to hold arguments for -m and -t
declare -a models
declare -a test_sets
seeds=(0 42 723)

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

if [ -z "${models}" ]; then
    echo "Please set -m"
    exit 1
fi

if [ -z "${test_sets}" ]; then
    echo "Please set -t"
    exit 1
fi

echo "Seeds: ${seeds[@]}"
echo "Models: ${models[@]}"
echo "Test sets: ${test_sets[@]}"


for model in "${models[@]}"; do
    for test_set in "${test_sets[@]}"; do
        for seed in "${seeds[@]}"; do
            echo ""
            echo "${model} --- ${test_set} --- ${seed}"
            echo ""

            python -m inference "${model}" \
                --input_file "${test_set}" \
                --batch_size 128 \
                --seed "${seed}" \
                --output_path "data/xquad_outputs" \
                --prompt_format "prompts/blank" \
                --src_key "instruction" --tgt_key "answer" \
                --stop "### Human:" "### Assistant:" "### Human" "### Assistant" \
                --n_gpus 4 --temperature 0.001

        done
    done
done
