"""
F1 vs latency trade-off for routing strategies (curated Q1--Q3, Context B).

Outputs:
  Preparation_of_Papers_for_IEEE_ACCESS/fig_routing_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_DIR = Path(__file__).resolve().parent.parent / "Preparation_of_Papers_for_IEEE_ACCESS"
OUT_PATH = PAPER_DIR / "fig_routing_tradeoff.png"

STRATEGIES = [
    {"name": "SLM only", "f1": 0.6101, "latency": 22.18, "color": "#2ca02c"},
    {"name": "Hybrid routing", "f1": 0.5667, "latency": 35.86, "color": "#ff7f0e"},
    {"name": "LLM only", "f1": 0.4548, "latency": 49.55, "color": "#1f77b4"},
]


def main() -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=150)

    for s in STRATEGIES:
        ax.scatter(
            s["latency"],
            s["f1"],
            s=220,
            c=s["color"],
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
            label=s["name"],
        )
        ax.annotate(
            f"{s['name']}\nF1={s['f1']:.4f}, {s['latency']:.2f}\,s",
            (s["latency"], s["f1"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            ha="left",
        )

    ax.set_xlabel("Latency per request (s, Context B concurrent load)")
    ax.set_ylabel("Token-F1 (curated Q1--Q3, $n{=}3$)")
    ax.set_title("Quality--latency trade-off: static vs.\ hybrid routing")
    ax.set_xlim(18, 54)
    ax.set_ylim(0.42, 0.64)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    plt.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
