#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=300GB
#SBATCH --time=23:59:00
#SBATCH --partition=lowprio


# Usage: sbatch scripts/slurm_train_small.sh <model_name_or_path> <train_dataset> <eval_dataset> <output_dir> <log_file>
# Example: sbatch scripts/slurm_train_small.sh "meta-llama/Meta-Llama-3-8B" "data/guanaco/guanaco_train_ml2.json" "data/guanaco/guanaco_test.json" "resources/models/llama_3_8b_ml2" "resources/models/logs/llama_3_8b_ml2.log"

BASE="/data/tkew/projects/multilingual-instruction-tuning" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

model_name_or_path=$1 # "meta-llama/Meta-Llama-3-8B"
train_dataset=$2 # "data/guanaco/guanaco_train_ml2.json"
eval_dataset=$3 # "data/guanaco/guanaco_test.json"
output_dir=$4 # resources/models/llama_3_8b_ml2
log_file=$5 # resources/models/logs/llama_3_8b_ml2.log

module purge
module load anaconda3 multigpu a100

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate vllm && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "${BASE}" && echo $(pwd) || exit 1

echo "*******************"
echo "BASE: ${BASE}"
echo "model_name_or_path: ${model_name_or_path}"
echo "train_dataset: ${train_dataset}"
echo "eval_dataset: ${eval_dataset}"
echo "output_dir: ${output_dir}"
echo "log_file: ${log_file}"
echo "*******************"


python sft_training.py \
    --model_name_or_path "${model_name_or_path}" \
    --train_dataset "${train_dataset}" \
    --eval_dataset "${eval_dataset}" \
    --output_dir "${output_dir}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 4 \
    --log_with "wandb" 2>&1 | tee "${log_file}"


echo "Training started..."