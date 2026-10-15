"""Train TinyStories GPT on one V100, validate, checkpoint, and generate.

Use --max-steps and --max-val-batches for a short workflow check. A complete
validation pass is required before reporting BPB against the full validation
text. Checkpoints are written under --run-dir so smoke runs stay separate.
"""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer

from dataset import TokenBlockDataset
from model import GPT, GPTConfig, set_seed


PROJECT_DIR = Path(__file__).resolve().parent


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(project_dir):
    config_path = project_dir / "config.json"
    tokenizer_path = project_dir / "tokenizer.json"
    metadata_path = project_dir / "data" / "processed" / "metadata.json"
    with config_path.open("r", encoding="utf-8") as source:
        config_dict = json.load(source)
    with metadata_path.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    config = GPTConfig(**config_dict)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_hash = file_sha256(tokenizer_path)
    metadata_hash = file_sha256(metadata_path)

    if metadata["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("tokenizer.json differs from the preprocessed data")
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from config.json")
    if tokenizer.token_to_id("<|endoftext|>") != metadata["eos_id"]:
        raise ValueError("EOS token differs from preprocessing metadata")
    if (metadata["vocab_size"] != config.vocab_size or
            metadata["block_size"] != config.block_size or
            metadata["token_dtype"] != "uint16_le"):
        raise ValueError("Preprocessed data and config.json do not match")

    processed = project_dir / "data" / "processed"
    train_path = processed / "train_tokens.bin"
    valid_path = processed / "valid_tokens.bin"
    for split, path in (("train", train_path), ("valid", valid_path)):
        expected_bytes = metadata[split]["tokens"] * 2
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Wrong size for {path}: expected {expected_bytes} bytes")

    train_data = TokenBlockDataset(train_path, config.block_size)
    valid_data = TokenBlockDataset(valid_path, config.block_size)
    if len(train_data) != metadata["train"]["blocks"]:
        raise ValueError("Training block count differs from metadata")
    if len(valid_data) != metadata["valid"]["blocks"]:
        raise ValueError("Validation block count differs from metadata")
    if len(train_data) == 0 or len(valid_data) == 0:
        raise ValueError("Training and validation datasets must be nonempty")
    return (config, config_dict, metadata, tokenizer, tokenizer_hash,
            metadata_hash, train_data, valid_data)


def make_train_loader(dataset, epoch, next_batch_index, batch_size, workers):
    # A separate generator makes the block order reproducible across restarts.
    order_generator = torch.Generator().manual_seed(42 + epoch)
    order = torch.randperm(len(dataset), generator=order_generator).tolist()
    start = next_batch_index * batch_size
    subset = Subset(dataset, order[start:])
    worker_generator = torch.Generator().manual_seed(10_000 + epoch)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, generator=worker_generator,
    )


@torch.no_grad()
def evaluate(model, valid_loader, device, total_bytes, max_batches=0):
    """Return token-weighted loss and full-set BPB when all blocks are used."""
    model.eval()
    total_nats = 0.0
    counted_tokens = 0
    start = time.monotonic()
    for batch_index, (x, y) in enumerate(valid_loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, targets=y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Validation loss is not finite")
        total_nats += loss.item() * y.numel()
        counted_tokens += y.numel()

    if counted_tokens == 0:
        raise ValueError("No validation blocks were evaluated")
    complete = not max_batches or max_batches >= len(valid_loader)
    avg_loss = total_nats / counted_tokens
    bpb = total_nats / (total_bytes * math.log(2)) if complete else None
    model.train()
    return {
        "val_loss": avg_loss,
        "val_bpb": bpb,
        "val_tokens": counted_tokens,
        "val_complete": complete,
        "val_seconds": time.monotonic() - start,
    }


@torch.no_grad()
def generate_story(model, tokenizer, prompt, new_tokens, eos_id, device):
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("Generation prompt produced no tokens")
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=new_tokens)[0].tolist()
    continuation = output[len(prompt_ids):]
    if eos_id in continuation:
        continuation = continuation[:continuation.index(eos_id)]
    return tokenizer.decode(prompt_ids + continuation, skip_special_tokens=True)


def atomic_torch_save(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def write_metric(path, event):
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False) + "\n")


def save_run_inputs(run_dir, config_path, tokenizer_path):
    """Keep the exact model configuration and tokenizer with this run."""
    for source in (config_path, tokenizer_path):
        destination = run_dir / source.name
        contents = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != contents:
                raise ValueError(f"Run directory has a different {source.name}")
        else:
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(contents)
            os.replace(temporary, destination)


def save_latest(path, model, optimizer, scheduler, scaler, state,
                config_dict, tokenizer_hash, metadata_hash, training_spec):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "state": state.copy(),
        "config": config_dict,
        "tokenizer_sha256": tokenizer_hash,
        "metadata_sha256": metadata_hash,
        "training_spec": training_spec,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }
    atomic_torch_save(payload, path)


def restore_latest(path, model, optimizer, scheduler, scaler,
                   config_dict, tokenizer_hash, metadata_hash, training_spec):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["config"] != config_dict or
            checkpoint["tokenizer_sha256"] != tokenizer_hash or
            checkpoint["metadata_sha256"] != metadata_hash or
            checkpoint["training_spec"] != training_spec):
        raise ValueError("Checkpoint config, tokenizer, dataset, or training setup has changed")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    return checkpoint["state"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_DIR / "checkpoints" / "main")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer-step attempts; 0 means all epochs")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0 evaluates all validation blocks and computes full BPB")
    parser.add_argument("--prompt", default="Once upon a time, a little girl")
    parser.add_argument("--generate-tokens", type=int, default=80)
    args = parser.parse_args()
    if (args.epochs < 1 or args.micro_batch_size < 1 or
            args.grad_accum_steps < 1 or args.val_batch_size < 1 or
            args.num_workers < 0 or args.log_every < 1 or
            args.save_every < 1 or args.val_every < 1 or
            args.max_steps < 0 or args.max_val_batches < 0 or
            args.generate_tokens < 0 or args.learning_rate <= 0 or
            args.min_learning_rate < 0 or
            args.min_learning_rate > args.learning_rate or
            args.weight_decay < 0 or args.grad_clip <= 0):
        parser.error("Invalid training arguments")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires one allocated CUDA GPU")
    device = torch.device("cuda")
    set_seed()
    (config, config_dict, metadata, tokenizer, tokenizer_hash,
     metadata_hash, train_data, valid_data) = load_inputs(PROJECT_DIR)
    eos_id = metadata["eos_id"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    latest_path = args.run_dir / "latest.pt"
    best_path = args.run_dir / "best.pt"
    samples_path = args.run_dir / "generations.txt"
    if args.resume:
        if args.resume.resolve() != latest_path.resolve():
            raise ValueError("Resume from the latest.pt in --run-dir")
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError("Run directory already has checkpoints; use --resume or a new --run-dir")
    save_run_inputs(args.run_dir, PROJECT_DIR / "config.json", PROJECT_DIR / "tokenizer.json")

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_data) / args.micro_batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    training_spec = {
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "val_batch_size": args.val_batch_size,
        "max_val_batches": args.max_val_batches,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "steps_per_epoch": steps_per_epoch,
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch,
        eta_min=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=2**14)
    state = {"epoch": 0, "next_batch_index": 0, "step": 0,
             "updates": 0, "tokens_seen": 0, "best_val_loss": float("inf")}
    if args.resume:
        state = restore_latest(
            args.resume, model, optimizer, scheduler, scaler,
            config_dict, tokenizer_hash, metadata_hash, training_spec,
        )
        if best_path.is_file():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            if (best_checkpoint["config"] != config_dict or
                    best_checkpoint["tokenizer_sha256"] != tokenizer_hash or
                    best_checkpoint["metadata_sha256"] != metadata_hash):
                raise ValueError("best.pt does not match this config, tokenizer, and data")
            # A validation checkpoint may have been written just before the
            # job stopped, while latest.pt still points to an older step.
            state["best_val_loss"] = min(
                state["best_val_loss"], best_checkpoint["val_loss"]
            )
        elif math.isfinite(state["best_val_loss"]):
            raise FileNotFoundError("The matching best.pt is missing from --run-dir")
        print(f"Resumed {args.resume}: step={state['step']} epoch={state['epoch']}", flush=True)

    valid_loader = DataLoader(
        valid_data, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("Train/valid blocks:", len(train_data), len(valid_data), flush=True)
    print("Effective batch:", args.micro_batch_size * args.grad_accum_steps,
          "epochs:", args.epochs, "max steps:", args.max_steps, flush=True)
    if args.max_val_batches:
        print("Partial validation: BPB will not be reported", flush=True)

    run_start = time.monotonic()
    while state["epoch"] < args.epochs:
        epoch = state["epoch"]
        batch_index = state["next_batch_index"]
        loader = make_train_loader(
            train_data, epoch, batch_index,
            args.micro_batch_size, args.num_workers,
        )
        iterator = iter(loader)
        model.train()
        while batch_index < batches_per_epoch:
            if args.max_steps and state["step"] >= args.max_steps:
                break
            group_size = min(args.grad_accum_steps, batches_per_epoch - batch_index)
            batches = [next(iterator) for _ in range(group_size)]
            group_tokens = sum(y.numel() for _, y in batches)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            step_start = time.monotonic()

            for x, y in batches:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, targets=y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at step {state['step'] + 1}")
                loss_sum += loss.item() * y.numel()
                scaler.scale(loss * (y.numel() / group_tokens)).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            if not skipped:
                scheduler.step()
                state["updates"] += 1

            batch_index += group_size
            state["step"] += 1
            state["tokens_seen"] += group_tokens
            if batch_index >= batches_per_epoch:
                state["epoch"] = epoch + 1
                state["next_batch_index"] = 0
            else:
                state["next_batch_index"] = batch_index

            step_seconds = time.monotonic() - step_start
            train_loss = loss_sum / group_tokens
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                event = {
                    "event": "train", "step": state["step"],
                    "epoch": epoch + 1, "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm), "loss_scale": scaler.get_scale(),
                    "skipped_update": skipped, "step_seconds": step_seconds,
                    "elapsed_seconds": time.monotonic() - run_start,
                    "tokens_seen": state["tokens_seen"],
                    "tokens_per_second": group_tokens / step_seconds,
                }
                write_metric(metrics_path, event)
                print("TRAIN", json.dumps(event), flush=True)

            stopping = (args.max_steps > 0 and state["step"] >= args.max_steps)
            epoch_done = state["epoch"] != epoch
            if state["step"] % args.val_every == 0 or stopping or epoch_done:
                result = evaluate(
                    model, valid_loader, device,
                    metadata["valid"]["raw_story_bytes"], args.max_val_batches,
                )
                print("VALID", json.dumps({"step": state["step"], **result}), flush=True)
                write_metric(metrics_path, {"event": "valid", "step": state["step"], **result})
                if result["val_loss"] < state["best_val_loss"]:
                    state["best_val_loss"] = result["val_loss"]
                    atomic_torch_save({
                        "model_state_dict": model.state_dict(),
                        "step": state["step"],
                        "val_loss": result["val_loss"],
                        "val_bpb": result["val_bpb"],
                        "val_complete": result["val_complete"],
                        "config": config_dict,
                        "tokenizer_sha256": tokenizer_hash,
                        "metadata_sha256": metadata_hash,
                    }, best_path)
                    print("BEST", best_path, "loss", state["best_val_loss"], flush=True)

                if args.generate_tokens:
                    story = generate_story(
                        model, tokenizer, args.prompt, args.generate_tokens,
                        eos_id, device,
                    )
                    with samples_path.open("a", encoding="utf-8") as output:
                        output.write(f"\nstep {state['step']}\n{story}\n")
                    print("STORY", repr(story), flush=True)

            if (state["step"] % args.save_every == 0 or stopping or epoch_done or
                    state["step"] % args.val_every == 0):
                save_latest(
                    latest_path, model, optimizer, scheduler, scaler,
                    state, config_dict, tokenizer_hash, metadata_hash,
                    training_spec,
                )
                print("CHECKPOINT", latest_path, "step", state["step"], flush=True)

            if stopping:
                break
        if args.max_steps and state["step"] >= args.max_steps:
            break

    print("FINISHED", json.dumps(state), "elapsed_seconds",
          time.monotonic() - run_start, flush=True)


if __name__ == "__main__":
    main()
