"""Encode a reproducible TinyStories subset into compact token streams.

This is a CPU preprocessing step. The validation split is encoded in full and
is never sampled into or mixed with the training split.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from train_tokenizer import EOS_TOKEN, iter_stories


PROJECT_DIR = Path(__file__).resolve().parent
TOKEN_DTYPE = np.dtype("<u2")  # 8192 IDs fit in unsigned 16-bit integers.
FLUSH_AFTER_TOKENS = 1_000_000


def count_stories(path):
    return sum(1 for _ in iter_stories(path, None))


def encode_split(path, destination, tokenizer, eos_id, block_size,
                 selected_indices=None, expected_stories=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    token_buffer = []
    stories_used = 0
    tokens_written = 0
    raw_story_bytes = 0
    stories_seen = 0

    with temporary.open("wb") as output:
        for story_index, story in enumerate(iter_stories(path, None)):
            stories_seen += 1
            if selected_indices is not None and story_index not in selected_indices:
                continue

            ids = tokenizer.encode(story).ids
            token_buffer.extend(ids)
            token_buffer.append(eos_id)
            stories_used += 1
            raw_story_bytes += len(story.encode("utf-8"))

            if len(token_buffer) >= FLUSH_AFTER_TOKENS:
                np.asarray(token_buffer, dtype=TOKEN_DTYPE).tofile(output)
                tokens_written += len(token_buffer)
                token_buffer.clear()

            if stories_used % 100_000 == 0:
                print(f"{destination.name}: encoded {stories_used:,} stories", flush=True)

        if token_buffer:
            np.asarray(token_buffer, dtype=TOKEN_DTYPE).tofile(output)
            tokens_written += len(token_buffer)

    if expected_stories is not None and stories_used != expected_stories:
        raise RuntimeError(
            f"Expected {expected_stories} stories from {path}, got {stories_used}"
        )
    if tokens_written <= block_size:
        raise RuntimeError(f"Too few tokens for a {block_size}-token block: {path}")
    if temporary.stat().st_size != tokens_written * TOKEN_DTYPE.itemsize:
        raise RuntimeError(f"Incomplete binary token file: {temporary}")

    os.replace(temporary, destination)
    blocks = (tokens_written - 1) // block_size
    return {
        "source_file": str(path.resolve()),
        "source_size_bytes": path.stat().st_size,
        "token_file": str(destination.resolve()),
        "stories_seen": stories_seen,
        "stories_used": stories_used,
        "tokens": tokens_written,
        "raw_story_bytes": raw_story_bytes,
        "blocks": blocks,
        "block_size": block_size,
        "usable_target_tokens": blocks * block_size,
        "unused_tail_tokens": (tokens_written - 1) % block_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path,
                        default=PROJECT_DIR / "data" / "TinyStories-train.txt")
    parser.add_argument("--valid-file", type=Path,
                        default=PROJECT_DIR / "data" / "TinyStories-valid.txt")
    parser.add_argument("--tokenizer", type=Path,
                        default=PROJECT_DIR / "tokenizer.json")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_DIR / "config.json")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "data" / "processed")
    parser.add_argument("--train-stories", type=int, default=800_000,
                        help="Number of randomly selected training stories; 0 uses all")
    args = parser.parse_args()

    if args.train_stories < 0:
        raise ValueError("--train-stories cannot be negative")
    for path in (args.train_file, args.valid_file, args.tokenizer, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)

    with args.config.open("r", encoding="utf-8") as source:
        config = json.load(source)
    block_size = config["block_size"]
    vocab_size = config["vocab_size"]
    if not 1 <= vocab_size <= 65_536:
        raise ValueError("This uint16 token format requires vocab_size <= 65536")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if tokenizer.get_vocab_size() != vocab_size or eos_id is None:
        raise ValueError("Tokenizer vocabulary or EOS token does not match config.json")

    print("Counting training stories for a seeded uniform sample...", flush=True)
    total_train_stories = count_stories(args.train_file)
    if args.train_stories and args.train_stories > total_train_stories:
        raise ValueError(
            f"Requested {args.train_stories} stories, only {total_train_stories} exist"
        )
    stories_to_use = args.train_stories or total_train_stories
    selected_indices = (
        set(random.Random(42).sample(range(total_train_stories), stories_to_use))
        if stories_to_use < total_train_stories else None
    )
    print(
        f"Training stories available: {total_train_stories:,}; "
        f"selected: {stories_to_use:,}; seed: 42",
        flush=True,
    )

    train_info = encode_split(
        args.train_file, args.output_dir / "train_tokens.bin", tokenizer,
        eos_id, block_size, selected_indices, stories_to_use,
    )
    if train_info["stories_seen"] != total_train_stories:
        raise RuntimeError("Training file changed between the count and encode passes")
    valid_info = encode_split(
        args.valid_file, args.output_dir / "valid_tokens.bin", tokenizer,
        eos_id, block_size,
    )

    metadata = {
        "dataset": "TinyStories",
        "train_selection": "uniform_without_replacement_original_file_order",
        "selection_seed": 42,
        "requested_train_stories": args.train_stories,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "token_dtype": "uint16_le",
        "vocab_size": vocab_size,
        "eos_token": EOS_TOKEN,
        "eos_id": eos_id,
        "block_size": block_size,
        "train": train_info,
        "valid": valid_info,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = args.output_dir / "metadata.json"
    temporary = metadata_file.with_name(metadata_file.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
    os.replace(temporary, metadata_file)

    print("Training stories:", train_info["stories_used"])
    print("Validation stories:", valid_info["stories_used"])
    print("Training tokens:", train_info["tokens"])
    print("Validation tokens:", valid_info["tokens"])
    print("Training blocks:", train_info["blocks"])
    print("Validation blocks:", valid_info["blocks"])
    print("Metadata:", metadata_file.resolve())
    print("PREPROCESSING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
