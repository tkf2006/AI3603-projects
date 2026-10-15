#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --partition=dgx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-best-%j.out
#SBATCH --error=logs/eval-best-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load miniconda3
source activate llm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p results
nvidia-smi
python eval_bpb.py \
  --checkpoint checkpoints/main/best.pt \
  --config checkpoints/main/config.json \
  --tokenizer checkpoints/main/tokenizer.json \
  --model_file model.py \
  --eval_data data/TinyStories-valid.txt \
  --output results/main_eval.json

# The supplied evaluator writes status=error to JSON but returns exit code 0.
python -c 'import json; r=json.load(open("results/main_eval.json", encoding="utf-8")); assert r["status"] == "ok", r.get("error")'
python generate_best.py \
  --run-dir checkpoints/main \
  --max-new-tokens 200 \
  --output results/main_best_story.txt
