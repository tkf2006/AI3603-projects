#!/bin/bash
#SBATCH --job-name=llm-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/data-%j.out
#SBATCH --error=logs/data-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python preprocess_data.py \
  --train-file data/TinyStories-train.txt \
  --valid-file data/TinyStories-valid.txt \
  --tokenizer tokenizer.json \
  --config config.json \
  --output-dir data/processed \
  --train-stories 800000
