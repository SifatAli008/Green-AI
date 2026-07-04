"""
Grouped bar chart for LLM-as-judge clinical scores (Table 3 data).

Outputs:
  Preparation_of_Papers_for_IEEE_ACCESS/fig_llm_judge_scores.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent.parent / "Preparation_of_Papers_for_IEEE_ACCESS"
OUT_PATH = PAPER_DIR / "fig_llm_judge_scores.png"

CONFIGS = ["SLM\n(No RAG)", "SLM\n+ RAG", "LLM\n(No RAG)", "LLM\n+ RAG"]
COLORS = ["#98df8a", "#2ca02c", "#aec7e8", "#1f77b4"]

SCORES = {
    "Correctness": [2.84, 2.86, 2.58, 2.69],
    "Completeness": [2.85, 2.92, 2.70, 2.83],
    "Clinical relevance": [3.17, 3.15, 3.00, 3.05],
}


def main() -> Path:
    dims = list(SCORES.keys())
    n_dims = len(dims)
    n_cfg = len(CONFIGS)
    x = np.arange(n_dims)
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)

    for i, (cfg, color) in enumerate(zip(CONFIGS, COLORS)):
        vals = [SCORES[d][i] for d in dims]
        offset = (i - (n_cfg - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=cfg.replace("\n", " "), color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Mean score (0--5 scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylim(2.4, 3.35)
    ax.set_title("LLM-as-judge clinical scores (Qwen2.5-7B-Instruct; $n{\\approx}8{,}950$--8,965 per configuration)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
