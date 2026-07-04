from __future__ import annotations

"""
Generate EIE Framework figures for documentation and paper.

Outputs (Preparation_of_Papers_for_IEEE_ACCESS/):
  fig_eie_pillars_comparison.png      — E / I / Ec bars (SLM vs LLM)
  fig_eie_energy_breakdown.png        — kWh/query and CO2e/M by strategy
  fig_eie_prototype_architecture.png  — prototype block diagram
  fig_eie_matrix_full_config.png      — 5×metrics normalized heatmap
  fig_eie_matrix_pairwise_kwh.png     — pairwise kWh/query ratio matrix
  fig_eie_matrix_grid_carbon.png      — grid intensity × strategy CO2e/M
  fig_eie_matrix_scaling.png          — query volume × strategy annual MWh
  fig_eie_matrix_rag_delta.png          — RAG vs NoRAG energy delta (%)
  fig_eie_matrix_eie_vs_clinical.png  — observables: EIE pillar vs clinical domain
  fig_eie_carbon_emissions.png        — kWh + CO2e/M dual axis (Pillar E)
  fig_eie_hardware_economy.png        — reference capex + electricity opex
  fig_eie_matrix_economy_full.png     — economy matrix M-14 heatmap
  fig_eie_framework_overview.png      — conceptual E / I / Ec framework diagram
"""

# Reference economy (measurement_config.py §7)
USD_PER_KWH = 0.12
CAPEX_SLM = 350.0
CAPEX_LLM = 1200.0
CAPEX_ROUTING = 1200.0
AMORT_YEARS = 3
QPD_1M = 1_000_000

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

PAPER_DIR = Path(__file__).resolve().parent.parent / "Preparation_of_Papers_for_IEEE_ACCESS__22_"
GRID_INTENSITY = 0.385

# All five measured strategies (measurement_config.py)
ALL_STRATEGIES = [
    "SLM_NoRAG", "SLM_RAG", "LLM_NoRAG", "LLM_RAG", "Routing_Hybrid",
]
ALL_LABELS = [
    "SLM\nNoRAG", "SLM\n+RAG", "LLM\nNoRAG", "LLM\n+RAG", "Routing\nHybrid",
]
KWH_ALL = [0.000052, 0.000052, 0.000111, 0.000112, 0.000101]
CO2_M_ALL = [19.83, 19.82, 42.35, 42.47, 38.19]
KWH_STD_ALL = [0.000011, 0.000013, 0.000024, 0.000023, 0.000021]

STRATEGIES = ["SLM_RAG", "LLM_RAG", "Routing"]
KWH = [0.000052, 0.000112, 0.000101]
CO2_M = [19.82, 42.47, 38.19]
KWH_STD = [0.000013, 0.000023, 0.000021]

SLM_INFRA = {
    "footprint_gb": 4,
    "latency_a_s": 2.3,
    "latency_b_s": 22.18,
    "throughput_tok_s": 119,
}
LLM_INFRA = {
    "footprint_gb": 14,
    "latency_a_s": 7.0,
    "latency_b_s": 49.55,
    "throughput_tok_s": 60,
}


def plot_eie_framework_overview() -> Path:
    """Conceptual EIE framework: three pillars reported separately from clinical quality."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(xy, w, h, text, color="#e8f4fc", ec="#1f4e79", fs=7.5, bold=False):
        p = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color,
            edgecolor=ec,
            linewidth=1.0,
        )
        ax.add_patch(p)
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            linespacing=1.25,
        )

    def arrow(p1, p2, style="->", color="#444"):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle=style,
                color=color,
                mutation_scale=11,
                linewidth=1.0,
            )
        )

    box((1.5, 8.6), 7.0, 0.9, "Clinical MQA workload\n(query stream)", "#fff8e1", "#b8860b", fs=7.5)
    box(
        (1.0, 6.8),
        8.0,
        1.3,
        "Instrumented RAG-CDS prototype\nRule-based router · SLM / LLM · FAISS+BM25 index",
        "#e8f5e9",
        "#2e7d32",
        fs=7.5,
    )
    box((0.8, 4.5), 8.4, 0.55, "EIE measurement plane (NVML telemetry, deployment logs)", "#eceff1", "#455a64", fs=7.5, bold=True)

    pillar_specs = [
        (1.0, 2.4, "Pillar E\nEnergy", "kWh/query\nCO₂e/M", "#bbdefb", "#1565c0"),
        (3.7, 2.4, "Pillar I\nInfrastructure", "VRAM · latency\nthroughput", "#c8e6c9", "#2e7d32"),
        (6.4, 2.4, "Pillar Ec\nEconomy", "electricity opex\nhardware capex", "#ffe0b2", "#ef6c00"),
    ]
    for x, y, title, detail, fill, edge in pillar_specs:
        box((x, y), 2.3, 1.55, f"{title}\n{detail}", fill, edge, fs=7.0)

    box((1.2, 0.35), 3.2, 0.85, "EIE results tables\n(Pillars E, I, Ec)", "#e3f2fd", "#1565c0", fs=7.0)
    box((5.6, 0.35), 3.2, 0.85, "Clinical quality tables\n(accuracy, faithfulness)", "#fce4ec", "#c2185b", fs=7.0)

    arrow((5.0, 8.6), (5.0, 8.1))
    arrow((5.0, 6.8), (5.0, 6.35))
    arrow((5.0, 5.05), (5.0, 4.5))
    arrow((2.15, 4.5), (2.15, 3.95))
    arrow((5.0, 4.5), (5.0, 3.95))
    arrow((7.85, 4.5), (7.85, 3.95))
    arrow((2.15, 2.4), (2.8, 1.2))
    arrow((7.85, 2.4), (7.2, 1.2))

    ax.plot([4.6, 5.4], [1.0, 1.0], color="#9e9e9e", linestyle="--", linewidth=0.9)
    ax.text(5.0, 1.15, "reported separately", ha="center", va="bottom", fontsize=6.5, color="#616161", style="italic")

    ax.text(5.0, 9.75, "EIE Framework", ha="center", fontsize=10, fontweight="bold", color="#1a237e")

    out = PAPER_DIR / "fig_eie_framework_overview.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    return out


def plot_pillars_comparison() -> Path:
    """Grouped comparison of normalized pillar metrics (SLM_RAG vs LLM_RAG)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=150)
    slm_c, llm_c = "#2ca02c", "#1f77b4"

    # E — Energy
    ax = axes[0]
    vals = [KWH[0] * 1e6, KWH[1] * 1e6]
    bars = ax.bar(["SLM+RAG", "LLM+RAG"], vals, color=[slm_c, llm_c], edgecolor="white")
    ax.set_ylabel("kWh / 1M queries")
    ax.set_title("Pillar E — Energy")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)

    # I — Infrastructure (Context A latency + footprint)
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [SLM_INFRA["latency_a_s"], LLM_INFRA["latency_a_s"]], w, label="Latency (s, Ctx A)", color=slm_c, alpha=0.85)
    ax.bar(x + w / 2, [SLM_INFRA["footprint_gb"], LLM_INFRA["footprint_gb"]], w, label="VRAM (GB)", color=llm_c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["SLM+RAG", "LLM+RAG"])
    ax.set_title("Pillar I — Infrastructure")
    ax.legend(fontsize=7, loc="upper left")

    # Ec — Economy (relative kWh proxy)
    ax = axes[2]
    rel = [1.0, KWH[1] / KWH[0]]
    bars = ax.bar(["SLM+RAG", "LLM+RAG"], rel, color=[slm_c, llm_c], edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Relative kWh/query (SLM = 1.0)")
    ax.set_title("Pillar Ec — Economy (energy proxy)")
    for b, v in zip(bars, rel):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}×", ha="center", va="bottom", fontsize=9)

    fig.suptitle("EIE Framework — Measured comparison (SLM+RAG vs LLM+RAG)", fontsize=11, y=1.02)
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_pillars_comparison.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_energy_breakdown() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
    colors = ["#98df8a", "#d62728", "#ff7f0e"]
    x = np.arange(len(STRATEGIES))

    ax1.bar(x, [k * 1e6 for k in KWH], color=colors, yerr=[s * 1e6 for s in KWH_STD], capsize=4, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(STRATEGIES, rotation=15, ha="right")
    ax1.set_ylabel("kWh / 1M queries (mean ± std)")
    ax1.set_title("GPU energy by deployment strategy (NVML, 10×20 trials)")

    ax2.bar(x, CO2_M, color=colors, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(STRATEGIES, rotation=15, ha="right")
    ax2.set_ylabel("kg CO₂e / 1M queries")
    ax2.set_title("Carbon at 0.385 kg CO₂e/kWh (U.S. reference)")

    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_energy_breakdown.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_prototype_architecture() -> Path:
    """Block diagram of the EIE-evaluated prototype."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(xy, w, h, text, color="#e8f4fc", ec="#1f77b4"):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=color, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", color="#444", mutation_scale=12, linewidth=1.2))

    box((0.3, 4.2), 1.8, 0.9, "Clinical query\n(Q1–Q12)", "#fff3cd", "#856404")
    box((2.5, 4.2), 2.0, 0.9, "Rule-based router\neval_routing.py", "#d4edda", "#155724")
    box((5.0, 4.8), 1.6, 0.7, "SLM path\nGemma-2-2B", "#c3e6cb", "#155724")
    box((5.0, 3.5), 1.6, 0.7, "LLM path\nLlama-2-7B", "#bee5eb", "#0c5460")
    box((7.0, 4.2), 2.2, 0.9, "Hybrid RAG\nFAISS+BM25", "#e2d5f1", "#4a235a")
    box((0.3, 2.2), 2.4, 0.9, "Corpus: 115 docs\n556 chunks", "#f8d7da", "#721c24")
    box((3.0, 2.2), 2.2, 0.9, "Embedder\nMiniLM-L6-v2", "#e8f4fc", "#1f77b4")
    box((5.5, 2.2), 1.8, 0.9, "FAISS\nIndexFlatIP", "#e8f4fc", "#1f77b4")
    box((7.6, 2.2), 1.8, 0.9, "BM25\nre-rank", "#e8f4fc", "#1f77b4")
    box((2.0, 0.5), 2.4, 0.9, "NVML telemetry\npynvml / T4", "#fff3cd", "#856404")
    box((5.0, 0.5), 2.4, 0.9, "EIE metrics\nE · I · Ec", "#d1ecf1", "#0c5460")
    box((7.8, 0.5), 1.6, 0.9, "Audit log\n(on-prem)", "#f5f5f5", "#333")

    arrow((2.1, 4.65), (2.5, 4.65))
    arrow((4.5, 4.85), (5.0, 5.0))
    arrow((4.5, 4.45), (5.0, 3.9))
    arrow((6.6, 4.65), (7.0, 4.65))
    arrow((2.1, 2.65), (3.0, 2.65))
    arrow((5.2, 2.65), (5.5, 2.65))
    arrow((7.3, 2.65), (7.6, 2.65))
    arrow((8.1, 4.2), (8.1, 3.1))
    arrow((3.2, 0.95), (5.0, 0.95))

    ax.text(5.0, 5.65, "EIE Prototype — Local clinical RAG-CDS pipeline", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.15, "Deployment: Kaggle T4 (eval) · target: on-premise workstation / edge GPU", ha="center", fontsize=8, color="#555")

    out = PAPER_DIR / "fig_eie_prototype_architecture.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _heatmap(ax, data, xlabels, ylabels, title, fmt=".2f", cmap="YlOrRd", vmin=None, vmax=None):
    if HAS_SNS:
        sns.heatmap(
            data, ax=ax, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=xlabels, yticklabels=ylabels,
            vmin=vmin, vmax=vmax, cbar_kws={"shrink": 0.85},
            linewidths=0.5, linecolor="white",
        )
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(xlabels)))
        ax.set_yticks(range(len(ylabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title, fontsize=10)


def plot_matrix_full_config() -> Path:
    """Matrix M-2: strategies × normalized EIE metrics (1 = highest burden)."""
    metrics = ["kWh/query\n(×10⁵)", "CO₂e/M\n(kg)", "kWh CV%\n(std/mean)", "L_A\n(s)", "L_B\n(s)", "VRAM\n(GB)"]
    # Raw: kWh×1e5, CO2_M, CV%, L_A, L_B, VRAM
    raw = np.array([
        [5.2, 19.83, 21.2, 2.3, 22.18, 4],
        [5.2, 19.82, 25.0, 2.3, 22.18, 4],
        [11.1, 42.35, 21.6, 7.0, 49.55, 14],
        [11.2, 42.47, 20.5, 7.0, 49.55, 14],
        [10.1, 38.19, 20.8, 3.8, 35.86, 14.37],  # L_A routing: weighted est.; L_B measured hybrid
    ])
    # Min-max normalize per column (1 = max burden; invert throughput would be separate)
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        norm[:, j] = (col - mn) / (mx - mn) if mx > mn else 0.5

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    _heatmap(ax, norm, metrics, ALL_LABELS, "Matrix M-2: Configuration × EIE metrics (normalized burden, 1 = highest)")
    fig.text(0.5, 0.01, "Raw measured values in EIE_FRAMEWORK.md Table M-2. Routing L_A ≈ 3.8 s (68%×2.3 + 24%×7 + 8%×7).", ha="center", fontsize=7, color="#555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = PAPER_DIR / "fig_eie_matrix_full_config.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_pairwise_kwh() -> Path:
    """Matrix M-3: pairwise kWh/query ratio (row ÷ col)."""
    kwh = np.array(KWH_ALL)
    n = len(kwh)
    ratio = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ratio[i, j] = kwh[i] / kwh[j] if kwh[j] else np.nan
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    short = [s.replace("_", " ") for s in ALL_STRATEGIES]
    _heatmap(ax, ratio, short, short, "Matrix M-3: Pairwise kWh/query ratio (row ÷ column)", fmt=".2f", cmap="RdYlGn_r", vmin=0.4, vmax=2.2)
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_pairwise_kwh.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_grid_carbon() -> Path:
    """Matrix M-6: grid intensity × strategy CO₂e/M."""
    grids = ["Iceland\n0.023", "UK\n0.233", "U.S.\n0.385", "EU avg\n0.450", "Coal-heavy\n0.700"]
    gamma = [0.023, 0.233, 0.385, 0.450, 0.700]
    kwh_m = np.array(KWH_ALL) * 1e6
    data = np.array([[k * g for g in gamma] for k in kwh_m])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _heatmap(ax, data, grids, ALL_LABELS, "Matrix M-6: CO₂e per 1M queries (kg) — grid × strategy", fmt=".1f", cmap="YlOrRd")
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_grid_carbon.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_scaling() -> Path:
    """Matrix M-7: daily query volume × strategy annual GPU MWh."""
    volumes = ["1K/day", "10K/day", "100K/day", "1M/day", "10M/day"]
    qpd = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    days = 365
    data = np.array([[k * q * days / 1000 for q in qpd] for k in KWH_ALL])  # MWh/year
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _heatmap(ax, data, volumes, ALL_LABELS, "Matrix M-7: Annual GPU energy (MWh/year) — volume × strategy", fmt=".1f", cmap="YlOrRd")
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_scaling.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_rag_delta() -> Path:
    """Matrix M-5: RAG vs NoRAG percent change on energy metrics."""
    metrics = ["kWh/query", "CO₂e/M", "kWh std"]
    slm_d = [
        100 * (KWH_ALL[1] - KWH_ALL[0]) / KWH_ALL[0],
        100 * (CO2_M_ALL[1] - CO2_M_ALL[0]) / CO2_M_ALL[0],
        100 * (KWH_STD_ALL[1] - KWH_STD_ALL[0]) / KWH_STD_ALL[0],
    ]
    llm_d = [
        100 * (KWH_ALL[3] - KWH_ALL[2]) / KWH_ALL[2],
        100 * (CO2_M_ALL[3] - CO2_M_ALL[2]) / CO2_M_ALL[2],
        100 * (KWH_STD_ALL[3] - KWH_STD_ALL[2]) / KWH_STD_ALL[2],
    ]
    data = np.array([slm_d, llm_d])
    fig, ax = plt.subplots(figsize=(5.5, 2.8), dpi=150)
    _heatmap(ax, data, metrics, ["SLM", "LLM"], "Matrix M-5: RAG effect on energy (% Δ vs NoRAG)", fmt="+.1f", cmap="RdBu_r", vmin=-5, vmax=5)
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_rag_delta.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_eie_vs_clinical() -> Path:
    """Matrix M-9: which reporting domain each observable belongs to."""
    observables = [
        "kWh/query", "CO₂e/M", "VRAM (GB)", "Latency", "Throughput",
        "MedQA acc.", "MMLU-Med acc.", "PubMedQA acc.", "token-F1 (curated)",
        "Faithfulness", "Recall@3", "LLM-judge score",
    ]
    domains = ["E", "Ec", "I", "I", "I", "Clinical", "Clinical", "Clinical", "Clinical", "Clinical", "Retrieval", "Clinical"]
    # 1 if observable maps to pillar/domain column
    cols = ["Pillar E", "Pillar I", "Pillar Ec", "Clinical", "Retrieval"]
    mat = np.zeros((len(observables), len(cols)))
    mapping = {"E": 0, "I": 1, "Ec": 2, "Clinical": 3, "Retrieval": 4}
    for i, d in enumerate(domains):
        if d == "E":
            mat[i, 0] = 1
        elif d == "Ec":
            mat[i, 2] = 1
        elif d == "I":
            mat[i, 1] = 1
        else:
            mat[i, mapping[d]] = 1
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    _heatmap(ax, mat, cols, observables, "Matrix M-9: Observable × reporting domain (1 = belongs)", fmt=".0f", cmap="Blues", vmin=0, vmax=1)
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_eie_vs_clinical.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_carbon_emissions() -> Path:
    """Pillar E: dual-axis kWh/M and CO2e/M — publication style (caption in paper)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig, ax1 = plt.subplots(figsize=(3.5, 2.65), dpi=300)
    x = np.arange(len(ALL_STRATEGIES))
    w = 0.55
    kwh_m = np.array(KWH_ALL) * 1e6
    bars = ax1.bar(
        x,
        kwh_m,
        w,
        color="#4a6fa5",
        edgecolor="white",
        linewidth=0.6,
        label="kWh / M queries",
        zorder=2,
    )
    ax1.set_ylabel("kWh / M queries")
    ax1.set_xlabel("Deployment strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("_", "\n") for s in ALL_STRATEGIES])
    ax1.set_ylim(0, max(kwh_m) * 1.18)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        CO2_M_ALL,
        "o-",
        color="#2d6a4f",
        linewidth=1.4,
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor="#1b4332",
        label=r"kg CO$_2$e / M queries",
        zorder=3,
    )
    ax2.set_ylabel(r"kg CO$_2$e / M queries")
    ax2.set_ylim(0, max(CO2_M_ALL) * 1.15)
    ax2.spines["top"].set_visible(False)
    for i, v in enumerate(CO2_M_ALL):
        ax2.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color="#1b4332")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.95, edgecolor="#ccc", fancybox=False)
    fig.tight_layout(pad=0.4)
    out = PAPER_DIR / "fig_eie_carbon_emissions.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    return out


def plot_hardware_economy() -> Path:
    """Pillar Ec: reference GPU capex and annual electricity @ 1M queries/day."""
    tiers = ["SLM tier\n(4 GB)", "LLM / Routing tier\n(14 GB)"]
    capex = [CAPEX_SLM, CAPEX_LLM]
    elec_yr_slm = KWH_ALL[1] * QPD_1M * 365 * USD_PER_KWH
    elec_yr_llm = KWH_ALL[3] * QPD_1M * 365 * USD_PER_KWH
    elec_yr_route = KWH_ALL[4] * QPD_1M * 365 * USD_PER_KWH
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
    ax1.bar(tiers, capex, color=["#2ca02c", "#1f77b4"], edgecolor="white")
    ax1.set_ylabel("Reference GPU capex (USD)")
    ax1.set_title("Hardware reference cost (Table Ec-3)")
    for i, v in enumerate(capex):
        ax1.text(i, v + 30, f"${v:.0f}", ha="center", fontsize=9)

    labels = ["SLM_RAG", "LLM_RAG", "Routing"]
    ax2.bar(labels, [elec_yr_slm, elec_yr_llm, elec_yr_route],
            color=["#98df8a", "#aec7e8", "#ffbb78"], edgecolor="white")
    ax2.set_ylabel("USD / year @ 1M queries/day")
    ax2.set_title(f"Electricity opex (${USD_PER_KWH}/kWh ref.; measured kWh)")
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_hardware_economy.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_matrix_economy_full() -> Path:
    """Matrix M-14: strategies × economy columns (normalized for heatmap)."""
    capex_yr = [CAPEX_SLM / AMORT_YEARS, CAPEX_SLM / AMORT_YEARS, CAPEX_LLM / AMORT_YEARS,
                CAPEX_LLM / AMORT_YEARS, CAPEX_ROUTING / AMORT_YEARS]
    elec_yr = [k * QPD_1M * 365 * USD_PER_KWH for k in KWH_ALL]
    tonnes = [c * QPD_1M * 365 / 1e6 for c in CO2_M_ALL]
    combined = [e + h for e, h in zip(elec_yr, capex_yr)]
    cols = ["CO₂e t/yr", "USD elec/yr", "USD hw/yr", "USD combined/yr"]
    raw = np.column_stack([tonnes, elec_yr, capex_yr, combined])
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        norm[:, j] = (col - mn) / (mx - mn) if mx > mn else 0.5
    annot = [[f"{raw[i,j]:.0f}" if j > 0 else f"{raw[i,j]:.2f}" for j in range(4)] for i in range(5)]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    if HAS_SNS:
        sns.heatmap(norm, ax=ax, annot=annot, fmt="", cmap="YlOrRd",
                    xticklabels=cols, yticklabels=ALL_LABELS, cbar_kws={"label": "Relative burden"})
    else:
        ax.imshow(norm, cmap="YlOrRd", aspect="auto")
    ax.set_title("Matrix M-14: Economy burden @ 1M queries/day")
    plt.tight_layout()
    out = PAPER_DIR / "fig_eie_matrix_economy_full.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fns = (
        plot_eie_framework_overview,
        plot_pillars_comparison,
        plot_energy_breakdown,
        plot_prototype_architecture,
        plot_matrix_full_config,
        plot_matrix_pairwise_kwh,
        plot_matrix_grid_carbon,
        plot_matrix_scaling,
        plot_matrix_rag_delta,
        plot_matrix_eie_vs_clinical,
        plot_carbon_emissions,
        plot_hardware_economy,
        plot_matrix_economy_full,
    )
    for fn in fns:
        p = fn()
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
