#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-full-%j.out
#SBATCH --error=logs/train-full-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi
resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args=(--resume checkpoints/main/latest.pt)
fi

python train.py \
  --run-dir checkpoints/main \
  --epochs 1 \
  --max-steps 0 \
  --micro-batch-size 2 \
  --grad-accum-steps 6 \
  --val-batch-size 8 \
  --num-workers 2 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --log-every 50 \
  --save-every 500 \
  --val-every 2000 \
  --max-val-batches 0 \
  --generate-tokens 80 \
  "${resume_args[@]}"
