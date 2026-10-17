"""Draw the architecture implemented in model.py for the course report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent / "results" / "figures" / "model_architecture.png"


def box(ax, x, y, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2, edgecolor="#354758", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.45)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color="#354758"))


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    box(ax, 2.0, 4.78, 4.0, 0.5, "Byte-level BPE  ·  8,192 tokens", "#e8f2fa", 12)
    box(ax, 2.0, 4.02, 4.0, 0.5, "Token + learned position embeddings", "#e8f2fa", 12)
    box(ax, 1.2, 1.95, 5.6, 1.62, "", "#f5f7fa")
    ax.text(4.0, 3.26, "Transformer block × 6", ha="center", fontsize=13,
            weight="bold", color="#243746")
    box(ax, 1.55, 2.72, 4.9, 0.42,
        "LayerNorm → 12 masked heads → projection → residual", "#deebe5", 10.5)
    box(ax, 1.55, 2.12, 4.9, 0.42,
        "LayerNorm → 4× GELU MLP → residual", "#e5e1f2", 10.5)
    box(ax, 2.0, 1.11, 4.0, 0.5, "Final LayerNorm + linear head", "#e8f2fa", 12)
    box(ax, 2.0, 0.16, 4.0, 0.5, "Next-token logits / greedy output", "#fae9dd", 12)
    for top, bottom in ((4.78, 4.52), (4.02, 3.57), (1.95, 1.61), (1.11, 0.66)):
        arrow(ax, (4.0, top - 0.03), (4.0, bottom + 0.03))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=230, facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
