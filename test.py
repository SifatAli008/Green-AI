"""Publication-quality Pillar E figure (kWh/M + CO2e/M) for IEEE Access."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = ["SLM_NoRAG", "SLM_RAG", "LLM_NoRAG", "LLM_RAG", "Routing_Hybrid"]
KWH_PER_QUERY = np.array([0.000052, 0.000052, 0.000111, 0.000112, 0.000101])
CO2E_PER_M = np.array([19.83, 19.82, 42.35, 42.47, 38.19])
KWH_PER_M = KWH_PER_QUERY * 1e6

OUT_PATH = (
    Path(__file__).resolve().parent
    / "Preparation_of_Papers_for_IEEE_ACCESS"
    / "fig_eie_carbon_emissions.png"
)

# IEEE-friendly: serif body, restrained palette
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    }
)

COLOR_BAR = "#4a6fa5"
COLOR_LINE = "#2d6a4f"
COLOR_MARKER = "#1b4332"


def _strategy_labels() -> list[str]:
    return [s.replace("_", "\n") for s in STRATEGIES]


def main() -> Path:
    x = np.arange(len(STRATEGIES))
    width = 0.55

    fig, ax1 = plt.subplots(figsize=(3.5, 2.65), dpi=300)

    bars = ax1.bar(
        x,
        KWH_PER_M,
        width=width,
        color=COLOR_BAR,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
        label="kWh / M queries",
    )

    ax1.set_ylabel("kWh / M queries")
    ax1.set_xlabel("Deployment strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(_strategy_labels())
    ax1.set_ylim(0, max(KWH_PER_M) * 1.18)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        CO2E_PER_M,
        color=COLOR_LINE,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=COLOR_MARKER,
        linewidth=1.4,
        zorder=3,
        label=r"kg CO$_2$e / M queries",
    )
    ax2.set_ylabel(r"kg CO$_2$e / M queries")
    ax2.set_ylim(0, max(CO2E_PER_M) * 1.15)
    ax2.spines["top"].set_visible(False)

    # Sparse value labels on line only (cleaner than labeling every bar)
    for xi, co2 in zip(x, CO2E_PER_M):
        ax2.text(xi, co2 + 1.0, f"{co2:.1f}", ha="center", va="bottom", fontsize=6.5, color=COLOR_MARKER)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=False,
        handlelength=1.6,
        borderpad=0.4,
    )

    fig.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
