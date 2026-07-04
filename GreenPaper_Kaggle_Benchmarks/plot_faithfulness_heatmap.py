"""
Generate faithfulness heatmap for paper Table 4 (RAG rows scored; NoRAG pending).

Outputs:
  Preparation_of_Papers_for_IEEE_ACCESS/fig_faithfulness_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT.parent / "Preparation_of_Papers_for_IEEE_ACCESS"
CSV_PATH = ROOT / "fathfullness" / "rerun" / "benchmark_results_all_predictions_combined_faithfulness_final.csv"
JSON_PATH = ROOT / "result" / "benchmark_results_all_predictions_combined_faithfulness.json"
OUT_PATH = PAPER_DIR / "fig_faithfulness_heatmap.png"

BENCHMARKS = ["medqa", "mmlu_med", "pubmedqa"]
CONFIG_ORDER = ["SLM_RAG", "LLM_RAG", "SLM_NoRAG", "LLM_NoRAG"]
CONFIG_LABELS = {
    "SLM_RAG": "SLM + RAG",
    "LLM_RAG": "LLM + RAG",
    "SLM_NoRAG": "SLM (No RAG)",
    "LLM_NoRAG": "LLM (No RAG)",
}
COL_LABELS = ["Faithfulness (%)", "Hallucination (%)", "MedQA", "MMLU-Med", "PubMedQA"]


def _load_aggregate_from_json() -> dict:
    if not JSON_PATH.is_file():
        return {}
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return data.get("aggregate_by_configuration") or {}


def _aggregate_from_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["faithfulness_primary"] = pd.to_numeric(df["faithfulness_primary"], errors="coerce")
    df["hallucination_rate_primary"] = pd.to_numeric(df["hallucination_rate_primary"], errors="coerce")
    rows = []
    for (model, bench), g in df.groupby(["model_name", "benchmark"], dropna=False):
        f = g["faithfulness_primary"].dropna()
        h = g["hallucination_rate_primary"].dropna()
        if len(f) == 0:
            continue
        rows.append(
            {
                "model_name": str(model),
                "benchmark": str(bench),
                "faithfulness_mean": float(f.mean()),
                "hallucination_mean": float(h.mean()) if len(h) else 100.0 - float(f.mean()),
                "n": int(len(f)),
            }
        )
    return pd.DataFrame(rows)


def _build_matrix(agg_csv: pd.DataFrame, agg_json: dict) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Return values matrix, mask (True=missing), annotation strings."""
    n_rows = len(CONFIG_ORDER)
    n_cols = len(COL_LABELS)
    values = np.full((n_rows, n_cols), np.nan)
    annot = [[""] * n_cols for _ in range(n_rows)]

    bench_map = {
        "medqa": "MedQA",
        "mmlu_med": "MMLU-Med",
        "pubmedqa": "PubMedQA",
    }
    col_for_bench = {v: i for i, v in enumerate(COL_LABELS) if v in bench_map.values()}

    for i, cfg in enumerate(CONFIG_ORDER):
        if cfg in agg_json:
            block = agg_json[cfg]
            faith = float(block.get("faithfulness_mean") or block.get("mean") or np.nan)
            hall = float(block.get("hallucination_rate_mean") or (100.0 - faith))
            values[i, 0] = faith
            values[i, 1] = hall
            annot[i][0] = f"{faith:.1f}"
            annot[i][1] = f"{hall:.1f}"
        sub = agg_csv[agg_csv["model_name"] == cfg]
        for _, row in sub.iterrows():
            bench = str(row["benchmark"])
            label = bench_map.get(bench)
            if label is None:
                continue
            j = COL_LABELS.index(label)
            v = float(row["faithfulness_mean"])
            values[i, j] = v
            annot[i][j] = f"{v:.1f}"

        if cfg.endswith("_NoRAG"):
            for j in range(n_cols):
                if np.isnan(values[i, j]):
                    annot[i][j] = "—"

    mask = np.isnan(values)
    return values, mask, annot


def main() -> Path:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Faithfulness CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    agg_csv = _aggregate_from_csv(df)
    agg_json = _load_aggregate_from_json()
    values, mask, annot = _build_matrix(agg_csv, agg_json)

    row_labels = [CONFIG_LABELS[c] for c in CONFIG_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)  # low=red (bad), high=green (good)

    sns.heatmap(
        values,
        ax=ax,
        mask=mask,
        annot=np.array(annot),
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=100,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Score (0–100)", "shrink": 0.85},
        xticklabels=COL_LABELS,
        yticklabels=row_labels,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(
        "Table 4 — Evidence faithfulness heatmap (Qwen/Qwen2.5-7B-Instruct judge)\n"
        "RAG rows scored (n=9,090 per model); No RAG pending extended run",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Configuration")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
