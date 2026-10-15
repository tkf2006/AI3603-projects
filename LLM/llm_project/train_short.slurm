#!/bin/bash
#SBATCH --job-name=llm-short
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/train-short-%j.out
#SBATCH --error=logs/train-short-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi
resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args=(--resume checkpoints/smoke/latest.pt)
fi
python train.py \
  --run-dir checkpoints/smoke \
  --micro-batch-size 2 \
  --grad-accum-steps 6 \
  --max-steps 20 \
  --log-every 1 \
  --save-every 10 \
  --val-every 10 \
  --max-val-batches 16 \
  --generate-tokens 40 \
  "${resume_args[@]}"
