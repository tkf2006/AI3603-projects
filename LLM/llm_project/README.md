# TinyStories LLM project

This folder contains the complete TinyStories language-model workflow: byte-level
BPE training, data preprocessing, a decoder-only Transformer, training and
validation, generation, and report figures. The validation text is not used for
tokenizer or model training. The public repository contains source code; datasets,
checkpoints, generated results, and the submission package stay local.

## Completed run

The final run used 800,000 stories sampled without replacement from the training
file (seed 42), one epoch, and a single Tesla V100. The 8,192-token BPE
tokenizer was trained on the first 200,000 training stories. The model has
55,504,896 parameters, six Transformer blocks, 12 attention heads, a 768-wide
hidden state, and a 512-token context window. Training took about 5 hours
38 minutes, including full validation and checkpoint I/O.

The course `eval_bpb.py` script loaded the submitted `model.py`, `config.json`,
`tokenizer.json`, and `best.pt` successfully. On the public TinyStories validation
set it reported an average loss of 1.6129 nats/token and BPB of 0.5615. These
are local validation results, not scores on the private grading set.

## Files

- `train_tokenizer.py`: streams stories, trains BPE, saves the tokenizer, and
  checks that the saved file can be loaded and used to encode/decode text.
- `train_tokenizer.slurm`: requests four CPU cores to run training on Pi 2.0.
- `requirements.txt`: Python dependency for this stage.
- `model.py`, `config.json`: GPT model and its matching 8,192-token configuration.
- `preprocess_data.py`, `preprocess_data.slurm`: CPU preprocessing of a seeded
  800,000-story training subset and the complete validation set.
- `dataset.py`: reads preprocessed tokens as non-overlapping next-token blocks.
- `train.py`, `train_short.slurm`, `train_full.slurm`: short and full GPU training.
- `eval_best.slurm`, `generate_best.py`: independent evaluation and generation.
- `plot_training_curves.py`, `plot_model_architecture.py`: create report figures
  from the saved metrics and model structure (requires Matplotlib).

## Run locally

The dataset currently sits one directory above this project folder. After
installing `tokenizers`, run from this folder:

```powershell
python -m pip install -r requirements.txt
python train_tokenizer.py --train-file ..\TinyStories-train.txt
```

The defaults use the first 200,000 stories and an 8,192-token vocabulary.
`--max-stories 0` trains on the entire training file. This may take much
longer. The script never reads the complete file into memory.

## Run on Pi 2.0

Upload `train_tokenizer.py` and `train_tokenizer.slurm` to
`~/llm_project/`. The data already belongs in
`~/llm_project/data/TinyStories-train.txt`. Ensure the `llm` Conda environment
has the `tokenizers` package. Create or install that environment within a CPU
compute allocation, following the course instructions. Then, from the Pi 2.0
login node, submit the CPU job:

```bash
cd ~/llm_project
sbatch train_tokenizer.slurm
squeue -u "$USER"
```

The job writes `tokenizer.json` in `~/llm_project/` and logs under `logs/`.
Its reported vocabulary size must match the future `config.json` value.

For model training or generation, load and use it like this:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
ids = tokenizer.encode("Once upon a time").ids
eos_id = tokenizer.token_to_id("<|endoftext|>")
ids.append(eos_id)
text = tokenizer.decode(ids[:-1])
```

## Prepare model data on Pi 2.0

Upload `preprocess_data.py`, `preprocess_data.slurm`, and `dataset.py` to
`~/llm_project/`. Keep the existing data and tokenizer under that directory.
From the login node, submit the CPU job:

```bash
cd ~/llm_project
mkdir -p logs
sbatch preprocess_data.slurm
```

The script counts the training stories, uses seed 42 to select exactly 800,000
uniformly without replacement, and encodes them in original file order. It
encodes every validation story separately. It writes little-endian uint16
streams to `data/processed/train_tokens.bin` and `valid_tokens.bin`; neither
file contains model weights. `data/processed/metadata.json` records story,
token, 512-token block, and source byte counts for the report. The validation
split must only be used to evaluate the model.

Each `TokenBlockDataset` item reads 513 adjacent token IDs and returns 512
input IDs and the next 512 IDs as targets. The final incomplete block is
ignored, matching the course evaluator's fixed-block layout. Training code
should shuffle training blocks and keep validation blocks in order.

## Train, validate, and generate on Pi 2.0

Upload `train.py` and `train_short.slurm` to `~/llm_project/`, alongside the
existing `model.py`, `dataset.py`, `config.json`, `tokenizer.json`, and
`data/processed/`. The training script checks their configuration, tokenizer
hash, and preprocessing metadata before allocating model weights. It calls
`set_seed()` before loading data or constructing the model.

From a **login** node, submit a short `dgx2` job; `sbatch` runs the Python
process on the allocated V100 compute node:

```bash
cd ~/llm_project
mkdir -p logs
sbatch train_short.slurm
squeue -u "$USER"
```

This job requests one 32 GB V100 and six CPU cores for at most 30 minutes.
It attempts 20 optimizer steps with FP16 autocast and gradient scaling,
accumulating six microbatches of two 512-token blocks each. The last
microbatch group of an epoch may be smaller. AdamW uses a cosine learning
rate schedule. The log contains `TRAIN` lines with step, elapsed time,
training loss, gradient norm, learning rate, and throughput. Validation and
greedy story generation run at steps 10 and 20. Short validation reads only
16 batches, so its loss is a pipeline check and `val_bpb` is `null`.

Look for `COMPLETED 0:0` in `sacct`, finite losses in the output log, and
files in `checkpoints/smoke/`: `best.pt`, `latest.pt`, `config.json`,
`tokenizer.json`, `metrics.jsonl`, and `generations.txt`. The `best.pt` file
uses a `model_state_dict` key accepted by the course evaluator. `latest.pt`
also contains optimizer, learning rate scheduler, gradient scaler, progress,
and random states for resumption. Short-job weights are only a workflow check.

If the short job stops after a saved step and needs to continue to step 20,
resubmit with the same arguments and run directory:

```bash
sbatch --export=ALL,RESUME=1 train_short.slurm
```

For a complete run later, use a **different** `--run-dir`, set
`--max-val-batches 0`, and choose the epoch count before starting. Complete
validation evaluates all 9,004 blocks and computes BPB from the validation
stories' original UTF-8 byte count. `best.pt` is updated only when validation
loss improves. A resumed run must use the same config, tokenizer, processed
data, epoch count, batch setup, and optimizer settings; its `--max-steps` may
be increased to let it run longer.

## Complete training and final validation

The successful 20-step V100 run used about 0.73 seconds per steady-state step.
With 337,948 training blocks and an effective batch of 12, one epoch needs
28,163 optimizer steps, approximately 5.7 hours of optimizer work. Full
validation, generation, and checkpoint I/O add time. This is an estimate,
not a guaranteed completion time.

Upload the current `train.py` and `train_full.slurm` to `~/llm_project/`.
The complete job uses the same 800,000-story training subset, one epoch,
full validation (`--max-val-batches 0`), and an independent
`checkpoints/main/` directory. It logs every 50 steps, saves `latest.pt`
every 500 steps, and evaluates all validation blocks every 2,000 steps
and at the end. Submit once from the login node:

```bash
cd ~/llm_project
mkdir -p logs
sbatch train_full.slurm
```

The Slurm time limit is 12 hours. If the job reaches the time limit before
`FINISHED`, resubmit the same file with `RESUME=1` after confirming
`checkpoints/main/latest.pt` exists:

```bash
sbatch --export=ALL,RESUME=1 train_full.slurm
```

Do not resume from `checkpoints/smoke/latest.pt`; its validation and training
settings differ. The final complete-validation `best.pt` is
`checkpoints/main/best.pt`, paired with the adjacent `config.json` and
`tokenizer.json`. `generations.txt` contains greedy samples after each
validation. Compare full validation loss and BPB against the assignment's
reference values as guides, not pass/fail thresholds.

For an independent final check with the supplied course evaluator, upload
`eval_bpb.py` from the assignment folder plus `generate_best.py` and
`eval_best.slurm`. After the training job has completed, submit:

```bash
sbatch eval_best.slurm
```

The evaluation job uses another allocated V100. It writes
`results/main_eval.json` with evaluator loss and BPB, and
`results/main_best_story.txt` with a greedy story generated specifically
from the best checkpoint. Check `status: ok` in the JSON. The grader's
private test set remains unavailable; this evaluates TinyStories-valid only.
