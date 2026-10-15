"""Generate a greedy story from a completed run's best validation checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/main"))
    parser.add_argument("--prompt", default="Once upon a time, a little girl")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/main_best_story.txt"))
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run story generation on an allocated GPU node")

    set_seed()
    config_path = args.run_dir / "config.json"
    tokenizer_path = args.run_dir / "tokenizer.json"
    checkpoint_path = args.run_dir / "best.pt"
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["config"] != config_dict or
            checkpoint["tokenizer_sha256"] != tokenizer_hash):
        raise ValueError("best.pt, config.json, and tokenizer.json do not match")
    if tokenizer.get_vocab_size() != config_dict["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match config.json")

    model = GPT(GPTConfig(**config_dict)).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("Prompt produced no token IDs")
    context = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    output = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
    continuation = output[len(prompt_ids):]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is not None and eos_id in continuation:
        continuation = continuation[:continuation.index(eos_id)]
    story = tokenizer.decode(prompt_ids + continuation, skip_special_tokens=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(story + "\n", encoding="utf-8")
    print("Best checkpoint step:", checkpoint["step"])
    print("Saved story:", args.output)
    print(story)


if __name__ == "__main__":
    main()
