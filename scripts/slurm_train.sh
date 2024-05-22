#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=512GB
#SBATCH --time=23:59:00
#SBATCH --partition=lowprio


# Usage: sbatch slurm_train.sh <model_name_or_path> <train_dataset> <eval_dataset> <output_dir> <log_file>
# Example: sbatch slurm_train.sh "meta-llama/Llama-2-70b-hf" "data/guanaco_train_ml2.json" "data/guanaco_eval.json" "resources/models/llama_2_70b_hf_ml2" "resources/models/logs/llama_2_70b_hf_ml2.log"
# sbatch scripts/slurm_train.sh "mistralai/Mixtral-8x7B-v0.1" "data/guanaco/guanaco_train_ml1.json" "data/guanaco/guanaco_test.json" "resources/models/mixtral_8x7b_ml1" "resources/models/logs/mixtral_8x7b_ml1.log"
# sbatch scripts/slurm_train.sh "mistralai/Mixtral-8x7B-v0.1" "data/guanaco/guanaco_train_ml2.json" "data/guanaco/guanaco_test.json" "resources/models/mixtral_8x7b_ml2" "resources/models/logs/mixtral_8x7b_ml2.log"
# sbatch scripts/slurm_train.sh "mistralai/Mixtral-8x7B-v0.1" "data/guanaco/guanaco_train_ml3.json" "data/guanaco/guanaco_test.json" "resources/models/mixtral_8x7b_ml3" "resources/models/logs/mixtral_8x7b_ml3.log"

# hardcoded defaults
BASE="/data/tkew/projects/multilingual-instruction-tuning" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

model_name_or_path=$1 # "meta-llama/Llama-2-70b-hf"
train_dataset=$2 # "data/guanaco/guanaco_train_ml2.json"
eval_dataset=$3 # "data/guanaco/guanaco_test.json"
output_dir=$4 # resources/models/llama_2_70b_hf_ml2
log_file=$5 # resources/models/logs/llama_2_70b_hf_ml2.log

module purge
module load anaconda3 multigpu a100

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
if [[ "$model_name_or_path" == "meta-llama"* ]]; then
    conda activate llama && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
elif [[ "$model_name_or_path" == "mistralai"* ]]; then
    conda activate ml_mixtral && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
else
    echo "Unknown model_name_or_path: $model_name_or_path"
    exit 1
fi
# conda activate vllm && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

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
    --max_steps 2000 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 \
    --log_with "wandb" 2>&1 | tee "${log_file}"


echo "Training started..."