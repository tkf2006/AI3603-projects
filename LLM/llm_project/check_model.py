"""Short model and tokenizer check; does not train or update any weights."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    with (project_dir / "config.json").open("r", encoding="utf-8") as source:
        config_dict = json.load(source)
    config = GPTConfig(**config_dict)

    tokenizer = Tokenizer.from_file(str(project_dir / "tokenizer.json"))
    actual_vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if actual_vocab_size != config.vocab_size or eos_id is None:
        raise RuntimeError(
            f"Tokenizer/config mismatch: {actual_vocab_size=}, "
            f"{config.vocab_size=}, {eos_id=}"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this job")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one allocated GPU")
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))
    else:
        device = torch.device("cpu")

    set_seed()
    print("Seed: 42")
    print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("Tokenizer vocab:", actual_vocab_size, "EOS ID:", eos_id)

    model = GPT(config).to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Parameters:", parameter_count)
    if not 50_000_000 <= parameter_count <= 120_000_000:
        raise RuntimeError("Parameter count is outside the shared assignment range")

    mask = model.blocks[0].att.heads[0].causal_mask
    if not torch.equal(mask, torch.ones_like(mask).tril()):
        raise RuntimeError("Causal mask is not lower triangular")

    tokens = torch.arange(1, 9, device=device).unsqueeze(0)
    targets = tokens.roll(shifts=-1, dims=1)
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp:
        logits, loss = model(tokens, targets)
    if logits.shape != (1, 8, config.vocab_size):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Loss is not finite: {loss.item()}")
    print("Logits shape:", tuple(logits.shape), "Loss:", loss.item())

    with torch.no_grad():
        generated = model.generate(tokens, max_new_tokens=2)
    if generated.shape != (1, 10):
        raise RuntimeError(f"Unexpected generated shape: {tuple(generated.shape)}")
    print("Generated shape:", tuple(generated.shape))
    print("MODEL CHECK PASSED; no weights were updated")


if __name__ == "__main__":
    main()
