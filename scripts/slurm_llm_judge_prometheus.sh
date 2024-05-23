#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=256GB
#SBATCH --time=23:59:00
#SBATCH --partition=lowprio

# Usage: 
# sbatch scripts/slurm_llm_judge_prometheus.sh -m <model_name_or_path> -t <test_datasets>
# Example:
# sbatch scripts/slurm_llm_judge_prometheus.sh -m llama_2_7b_hf_ml2_merged -l de en -t true -limit 300

# hardcoded defaults
BASE="/data/tkew/projects/multilingual-instruction-tuning" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

module purge
module load anaconda3 multigpu cuda/12.2.1
echo $(ml)

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate prometheus && echo "CONDA ENV: $CONDA_DEFAULT_ENV" # for llama2/falcon models!


cd "${BASE}" && echo $(pwd) || exit 1

bash scripts/run_llm_judge_prometheus.sh "$@"