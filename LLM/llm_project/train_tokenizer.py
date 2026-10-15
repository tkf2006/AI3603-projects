"""Train a byte-level BPE tokenizer on TinyStories.

The saved tokenizer.json is readable by tokenizers.Tokenizer.from_file(),
which is the format required by the course evaluation script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


EOS_TOKEN = "<|endoftext|>"
PROJECT_DIR = Path(__file__).resolve().parent


def iter_stories(path: Path, max_stories: int | None) -> Iterator[str]:
    """Yield stories separated by EOS_TOKEN without loading the full file."""
    count = 0
    pieces: list[str] = []

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            fragments = line.split(EOS_TOKEN)
            for index, fragment in enumerate(fragments):
                pieces.append(fragment)
                if index == len(fragments) - 1:
                    continue

                story = "".join(pieces).strip()
                pieces.clear()
                if story:
                    yield story
                    count += 1
                    if max_stories is not None and count >= max_stories:
                        return

    story = "".join(pieces).strip()
    if story and (max_stories is None or count < max_stories):
        yield story


def encode_story(tokenizer: Tokenizer, story: str, append_eos: bool = True) -> list[int]:
    """Encode one story, optionally ending it with the course EOS token."""
    ids = tokenizer.encode(story).ids
    if append_eos:
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        if eos_id is None:
            raise ValueError(f"Tokenizer is missing {EOS_TOKEN}")
        ids.append(eos_id)
    return ids


def train_bpe(
    train_file: Path,
    output_file: Path,
    vocab_size: int = 8192,
    max_stories: int | None = 200_000,
    min_frequency: int = 2,
) -> tuple[Tokenizer, int]:
    """Train and save a byte-level BPE tokenizer; return it and corpus size."""
    if not train_file.is_file():
        raise FileNotFoundError(f"Training text not found: {train_file}")
    if vocab_size < 257:
        raise ValueError("vocab_size must allow 256 byte tokens plus the EOS token")
    if max_stories is not None and max_stories < 1:
        raise ValueError("max_stories must be positive or None")
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    count = 0

    def corpus() -> Iterator[str]:
        nonlocal count
        for story in iter_stories(train_file, max_stories):
            count += 1
            yield story

    if max_stories is None:
        tokenizer.train_from_iterator(corpus(), trainer=trainer)
    else:
        tokenizer.train_from_iterator(corpus(), trainer=trainer, length=max_stories)

    if count == 0:
        raise ValueError("No stories found in the training text")
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(
            f"Expected {vocab_size} tokens, got {tokenizer.get_vocab_size()}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_file))
    return tokenizer, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=PROJECT_DIR / "data" / "TinyStories-train.txt",
        help="TinyStories training text, never the validation text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "tokenizer.json",
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--max-stories",
        type=int,
        default=200_000,
        help="Stories used to train BPE; 0 means the entire training file",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    story_limit = None if args.max_stories == 0 else args.max_stories
    _, story_count = train_bpe(
        args.train_file,
        args.output,
        vocab_size=args.vocab_size,
        max_stories=story_limit,
        min_frequency=args.min_frequency,
    )

    loaded = Tokenizer.from_file(str(args.output))
    sample = "Once upon a time, a little fox found a key.\n"
    encoded = encode_story(loaded, sample)
    if loaded.decode(encoded[:-1]) != sample:
        raise RuntimeError("Tokenizer failed the encode/decode round trip")

    print(f"Stories used: {story_count}")
    print(f"Vocabulary size: {loaded.get_vocab_size()}")
    print(f"EOS token ID: {loaded.token_to_id(EOS_TOKEN)}")
    print(f"Sample token IDs: {encoded[:20]}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
