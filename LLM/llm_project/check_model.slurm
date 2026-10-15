#!/bin/bash
#SBATCH --job-name=llm-check
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/model-check-%j.out
#SBATCH --error=logs/model-check-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi
python check_model.py --device cuda
