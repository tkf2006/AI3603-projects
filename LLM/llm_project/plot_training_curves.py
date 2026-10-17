"""Plot report-ready training curves from the main run's metrics.jsonl.

Run from any directory with ``python plot_training_curves.py``. The script
does not modify the metrics or model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def read_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

    train = [row for row in records if row.get("event") == "train"]
    valid = [row for row in records if row.get("event") == "valid"]
    if not train or not valid:
        raise ValueError("Both train and valid events are required")
    for event, rows, values in (
        ("train", train, ("train_loss", "lr")),
        ("valid", valid, ("val_loss", "val_bpb")),
    ):
        steps = [row["step"] for row in rows]
        if any(left >= right for left, right in zip(steps, steps[1:])):
            raise ValueError(f"{event} steps are not strictly increasing")
        for row in rows:
            for name in values:
                if not math.isfinite(row[name]):
                    raise ValueError(f"Non-finite {event} {name} at step {row['step']}")
    if any(not row.get("val_complete") for row in valid):
        raise ValueError("A validation event is not a full validation pass")
    return train, valid


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=float)
    return np.convolve(values, kernel, mode="same") / np.convolve(
        np.ones_like(values), kernel, mode="same"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path,
        default=PROJECT_DIR / "checkpoints" / "main" / "metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "results" / "figures"
    )
    args = parser.parse_args()
    train, valid = read_metrics(args.metrics)

    train_steps = np.array([row["step"] for row in train])
    train_loss = np.array([row["train_loss"] for row in train])
    learning_rate = np.array([row["lr"] for row in train])
    valid_steps = np.array([row["step"] for row in valid])
    valid_loss = np.array([row["val_loss"] for row in valid])
    best_index = int(np.argmin(valid_loss))
    best_step = int(valid_steps[best_index])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(8.5, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    axes = (
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )
    train_color = "#246a9b"
    valid_color = "#d06a30"

    axes[0].plot(train_steps, train_loss, color=train_color, alpha=0.27,
                 linewidth=0.9, label="Logged loss (every 50 steps)")
    axes[0].plot(train_steps, moving_average(train_loss, 11), color=train_color,
                 linewidth=2.0, label="11-point moving average")
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Cross-entropy (nats/token)")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(valid_steps, valid_loss, color=valid_color, marker="o",
                 markersize=3.6, linewidth=1.9, label="Full validation")
    axes[1].scatter([best_step], [valid_loss[best_index]], s=45,
                    color="#9c3232", zorder=3)
    axes[1].annotate(
        f"best: {valid_loss[best_index]:.4f}\nstep {best_step:,}",
        (best_step, valid_loss[best_index]), xytext=(-9, 15),
        textcoords="offset points", ha="right", va="bottom", fontsize=9,
    )
    axes[1].set_title("Validation loss")
    axes[1].set_ylabel("Cross-entropy (nats/token)")

    axes[2].plot(train_steps, learning_rate * 1e4, color="#4a8062",
                 linewidth=2.0)
    axes[2].set_title("Learning rate")
    axes[2].set_ylabel("Learning rate (×10⁻⁴)")

    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_xlim(0, max(best_step, int(train_steps[-1])) * 1.02)
        ax.grid(alpha=0.2, linewidth=0.6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "training_curves.png"
    svg = args.output_dir / "training_curves.svg"
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    print(f"Training points: {len(train)}; full validation points: {len(valid)}")
    print(f"Best validation: step={best_step}, loss={valid_loss[best_index]:.6f}")
    print(f"Saved: {png}")
    print(f"Saved: {svg}")


if __name__ == "__main__":
    main()
