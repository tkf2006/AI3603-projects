#!/bin/bash
#SBATCH --job-name=llm-bpe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/bpe-%j.out
#SBATCH --error=logs/bpe-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python train_tokenizer.py \
  --train-file data/TinyStories-train.txt \
  --output tokenizer.json \
  --vocab-size 8192 \
  --max-stories 200000
