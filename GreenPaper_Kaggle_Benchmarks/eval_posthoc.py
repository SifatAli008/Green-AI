"""
Build faculty-review paper tables from benchmark + post-hoc JSON artifacts.

Numbered results tables (see ``current plan on faculty reviwe.md``):
  1  – Retrieval (Recall@1/3/5/10, MRR) by RAG configuration
  2  – Answer quality (F1, BERTScore, ROUGE-L, accuracy mean ± std) by configuration
  3  – LLM-as-judge (correctness, completeness, clinical_relevance)
  4  – Faithfulness / hallucination rate (RAG and NoRAG, comparable grounding)
  S1 – Internal BERTScore (optional baseline model rows from predictions)
  S2 – Literature comparison (MedQA / PubMedQA vs internal metrics)
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

TABLE_1_RETRIEVAL = "Table 1 – Retrieval Performance"
TABLE_2_ANSWER_QUALITY = "Table 2 – Answer Quality (F1, BERTScore, lexical)"
TABLE_3_LLM_JUDGE = "Table 3 – LLM-as-Judge Clinical Scores"
TABLE_4_FAITHFULNESS = "Table 4 – Faithfulness and Hallucination"


def load_predictions_payload(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.get("rows") or [])
    return data, rows


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if v == v]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def _fmt_mean_std(mean: Optional[float], std: Optional[float], digits: int = 3) -> str:
    if mean is None or mean != mean:
        return "—"
    if std is None or std != std or std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _accuracy_std_from_predictions(
    predictions_json: str,
    *,
    benchmark: str,
    configuration: str,
) -> Optional[float]:
    """Per-item sample std of binary correctness (MCQ or PubMedQA label)."""
    if not predictions_json or not os.path.isfile(predictions_json):
        return None
    _, rows = load_predictions_payload(predictions_json)
    bits: List[float] = []
    for r in rows:
        if str(r.get("benchmark") or "") != benchmark:
            continue
        if str(r.get("model_name") or "") != configuration:
            continue
        if benchmark in ("medqa", "mmlu_med", "medmcqa"):
            ok = r.get("mcq_correct")
        else:
            ok = r.get("label_correct")
        if ok in (True, False, 1, 0, 1.0, 0.0):
            bits.append(1.0 if ok in (True, 1, 1.0) else 0.0)
    if len(bits) < 2:
        return 0.0 if bits else None
    return statistics.stdev(bits)


def table_x_retrieval(predictions_json: str) -> Dict[str, Any]:
    """Table 1 – Recall@k and MRR from RAG prediction rows."""
    _, rows = load_predictions_payload(predictions_json)
    rag_rows = [r for r in rows if r.get("rag_flag")]
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for r in rag_rows:
        if not r.get("retrieval_evaluable"):
            continue
        m = str(r.get("model_name") or "unknown")
        by_model.setdefault(m, {"mrr": [], "r1": [], "r3": [], "r5": [], "r10": []})
        if r.get("mrr") == r.get("mrr"):
            by_model[m]["mrr"].append(float(r["mrr"]))
        for key, bucket in (
            ("recall_at_1", "r1"),
            ("recall_at_3", "r3"),
            ("recall_at_5", "r5"),
            ("recall_at_10", "r10"),
        ):
            if key in r and r[key] == r[key]:
                by_model[m][bucket].append(float(r[key]))

    table_rows: List[Dict[str, Any]] = []
    for mname in sorted(by_model.keys()):
        b = by_model[mname]
        m1, s1 = _mean_std(b["r1"])
        m3, s3 = _mean_std(b["r3"])
        m5, s5 = _mean_std(b["r5"])
        m10, s10 = _mean_std(b["r10"])
        mm, sm = _mean_std(b["mrr"])
        table_rows.append(
            {
                "configuration": mname,
                "recall_at_1": m1,
                "recall_at_1_std": s1,
                "recall_at_1_display": _fmt_mean_std(m1, s1),
                "recall_at_3": m3,
                "recall_at_3_std": s3,
                "recall_at_3_display": _fmt_mean_std(m3, s3),
                "recall_at_5": m5,
                "recall_at_5_std": s5,
                "recall_at_5_display": _fmt_mean_std(m5, s5),
                "recall_at_10": m10,
                "recall_at_10_std": s10,
                "recall_at_10_display": _fmt_mean_std(m10, s10),
                "mrr_mean": mm,
                "mrr_std": sm,
                "mrr_display": _fmt_mean_std(mm, sm),
                "n_rows": max(len(b["r3"]), len(b["mrr"])),
            }
        )
    return {"title": TABLE_1_RETRIEVAL, "rows": table_rows}


def table_y_answer_quality(
    benchmark_json: str,
    predictions_json: str = "",
) -> Dict[str, Any]:
    """Table 2 – F1 / BERTScore / lexical metrics from ``eval_benchmarks`` aggregate JSON."""
    agg = load_json(benchmark_json)
    table_rows: List[Dict[str, Any]] = []
    for bench_name, bench in (agg.get("benchmarks") or {}).items():
        results = bench.get("results") or {}
        if not isinstance(results, dict):
            continue
        for cfg, metrics in sorted(results.items()):
            if not isinstance(metrics, dict):
                continue
            row: Dict[str, Any] = {
                "benchmark": bench_name,
                "configuration": cfg,
                "metric_type": metrics.get("metric"),
                "n": metrics.get("n"),
            }
            if metrics.get("metric") == "accuracy":
                row["accuracy_mean"] = metrics.get("mean")
                row["accuracy_std"] = metrics.get("std")
                row["f1_mean"] = metrics.get("f1_mean")
                row["f1_std"] = metrics.get("f1_std")
                if row.get("accuracy_std") is None:
                    row["accuracy_std"] = _accuracy_std_from_predictions(
                        predictions_json,
                        benchmark=bench_name,
                        configuration=cfg,
                    )
            elif metrics.get("metric") == "free_text":
                for key in (
                    "f1_mean",
                    "f1_std",
                    "bertscore_f1_mean",
                    "rougeL_mean",
                    "bleu_mean",
                    "meteor_mean",
                    "pubmedqa_label_accuracy_mean",
                    "pubmedqa_label_accuracy_std",
                ):
                    if key in metrics:
                        row[key.replace("_mean", "")] = metrics[key]
                if row.get("pubmedqa_label_accuracy") is not None and row.get(
                    "pubmedqa_label_accuracy_std"
                ) is None:
                    row["pubmedqa_label_accuracy_std"] = _accuracy_std_from_predictions(
                        predictions_json,
                        benchmark=bench_name,
                        configuration=cfg,
                    )
            table_rows.append(row)

    by_cfg: Dict[str, Dict[str, Any]] = {}
    for r in table_rows:
        cfg = str(r["configuration"])
        by_cfg.setdefault(cfg, {"configuration": cfg, "benchmarks": []})
        by_cfg[cfg]["benchmarks"].append(r)

    summary: List[Dict[str, Any]] = []
    for cfg, block in sorted(by_cfg.items()):
        f1s = [b.get("f1_mean") or b.get("f1") for b in block["benchmarks"] if b.get("f1_mean") is not None]
        berts = [
            b.get("bertscore_f1_mean") or b.get("bertscore_f1")
            for b in block["benchmarks"]
            if b.get("bertscore_f1_mean") is not None or b.get("bertscore_f1") is not None
        ]
        accs = [b.get("accuracy_mean") for b in block["benchmarks"] if b.get("accuracy_mean") is not None]
        summary.append(
            {
                "configuration": cfg,
                "mean_f1_across_benchmarks": statistics.mean(f1s) if f1s else None,
                "mean_bertscore_f1_across_benchmarks": statistics.mean(berts) if berts else None,
                "mean_accuracy_across_mcq_benchmarks": statistics.mean(accs) if accs else None,
                "per_benchmark": block["benchmarks"],
            }
        )

    return {
        "title": TABLE_2_ANSWER_QUALITY,
        "rows_detail": table_rows,
        "rows_by_configuration": summary,
    }


def table_z_llm_judge(judge_json: str) -> Dict[str, Any]:
    """Table 3 – LLM-as-judge clinical scores."""
    data = load_json(judge_json)
    by_model = data.get("aggregate_by_model") or {}
    rows: List[Dict[str, Any]] = []
    for cfg, scores in sorted(by_model.items()):
        if not isinstance(scores, dict):
            continue
        rows.append(
            {
                "configuration": cfg,
                "correctness": scores.get("correctness"),
                "correctness_std": scores.get("correctness_std"),
                "completeness": scores.get("completeness"),
                "completeness_std": scores.get("completeness_std"),
                "clinical_relevance": scores.get("clinical_relevance"),
                "clinical_relevance_std": scores.get("clinical_relevance_std"),
                "n_judged": scores.get("n_judged"),
            }
        )
    return {"title": TABLE_3_LLM_JUDGE, "rows": rows}


def table_h_faithfulness(faithfulness_json: str) -> Dict[str, Any]:
    """Table 4 – Faithfulness (%) and hallucination rate (%)."""
    data = load_json(faithfulness_json)
    by_cfg = data.get("aggregate_by_configuration") or {}
    rows: List[Dict[str, Any]] = []
    if by_cfg:
        for cfg, block in sorted(by_cfg.items()):
            if not isinstance(block, dict):
                continue
            rows.append(
                {
                    "configuration": cfg,
                    "faithfulness_pct": block.get("faithfulness_mean"),
                    "faithfulness_std": block.get("faithfulness_std"),
                    "hallucination_rate_pct": block.get("hallucination_rate_mean"),
                    "n": block.get("n"),
                    "backend": block.get("backend"),
                }
            )
    else:
        summ = data.get("summary_primary") or {}
        rows.append(
            {
                "configuration": "(all rows pooled)",
                "faithfulness_pct": summ.get("mean"),
                "faithfulness_std": summ.get("std"),
                "hallucination_rate_pct": (100.0 - summ["mean"]) if summ.get("mean") == summ.get("mean") else None,
                "n": summ.get("n"),
                "backend": (data.get("meta") or {}).get("backend"),
            }
        )
    return {"title": TABLE_4_FAITHFULNESS, "rows": rows}


def table_s1_internal_bertscore(
    benchmark_json: str,
    predictions_json: str = "",
) -> Dict[str, Any]:
    """Table S1 – BERTScore F1 by configuration (and optional inference_model_id)."""
    rows_out: List[Dict[str, Any]] = []
    agg = load_json(benchmark_json)
    seen_cfg: set[str] = set()
    for bench_name, bench in (agg.get("benchmarks") or {}).items():
        for cfg, metrics in sorted((bench.get("results") or {}).items()):
            if cfg in seen_cfg:
                continue
            if isinstance(metrics, dict) and metrics.get("bertscore_f1_mean") == metrics.get("bertscore_f1_mean"):
                seen_cfg.add(cfg)
                rows_out.append(
                    {
                        "configuration": cfg,
                        "bertscore_f1_mean": metrics.get("bertscore_f1_mean"),
                        "bertscore_f1_ci_lower": metrics.get("bertscore_f1_ci_lower"),
                        "bertscore_f1_ci_upper": metrics.get("bertscore_f1_ci_upper"),
                        "source_benchmark": bench_name,
                    }
                )

    if predictions_json and os.path.isfile(predictions_json):
        _, rows = load_predictions_payload(predictions_json)
        by_mid: Dict[str, List[float]] = {}
        for r in rows:
            mid = str(r.get("inference_model_id") or r.get("slm_model_id") or r.get("llm_model_id") or "")
            if not mid:
                continue
            bf = r.get("bertscore_f1") or r.get("token_f1")
            if isinstance(bf, (int, float)) and bf == bf:
                by_mid.setdefault(mid, []).append(float(bf))
        for mid, vals in sorted(by_mid.items()):
            m, s = _mean_std(vals)
            rows_out.append(
                {
                    "configuration": f"model_id:{mid}",
                    "bertscore_f1_mean": m,
                    "bertscore_f1_std": s,
                    "n_rows": len(vals),
                    "source": "predictions_per_row",
                }
            )

    return {"title": "Table S1 – Internal BERTScore Comparison", "rows": rows_out}


def table_s2_literature_comparison(
    literature_json: str,
    benchmark_json: str,
) -> Dict[str, Any]:
    """Table S2 – Published MedQA/PubMedQA vs this system's internal metrics."""
    lit = load_json(literature_json)
    internal = table_y_answer_quality(benchmark_json)
    internal_rows = internal.get("rows_by_configuration") or []

    def _internal_pubmedqa_acc(cfg: str) -> Optional[float]:
        for block in internal_rows:
            if block.get("configuration") != cfg:
                continue
            for b in block.get("per_benchmark") or []:
                if b.get("benchmark") == "pubmedqa":
                    return b.get("pubmedqa_label_accuracy") or b.get("pubmedqa_label_accuracy_mean")
        return None

    def _internal_medqa_acc(cfg: str) -> Optional[float]:
        for block in internal_rows:
            if block.get("configuration") != cfg:
                continue
            for b in block.get("per_benchmark") or []:
                if b.get("benchmark") in ("medqa", "mmlu_med", "medmcqa"):
                    return b.get("accuracy_mean")
        return None

    rows: List[Dict[str, Any]] = []
    for m in lit.get("models") or []:
        if not isinstance(m, dict):
            continue
        rows.append(
            {
                "model": m.get("name"),
                "medqa_literature": m.get("medqa_accuracy"),
                "pubmedqa_literature": m.get("pubmedqa_accuracy"),
                "notes": m.get("notes"),
                "source": "literature",
            }
        )

    for cfg in ("SLM_RAG", "LLM_RAG", "SLM_NoRAG", "LLM_NoRAG"):
        label = (lit.get("internal_system_labels") or {}).get(cfg, cfg)
        rows.append(
            {
                "model": label,
                "medqa_internal_accuracy": _internal_medqa_acc(cfg),
                "pubmedqa_internal_label_accuracy": _internal_pubmedqa_acc(cfg),
                "bertscore_f1_internal": next(
                    (b.get("mean_bertscore_f1_across_benchmarks") for b in internal_rows if b.get("configuration") == cfg),
                    None,
                ),
                "source": "this_work",
            }
        )

    return {
        "title": "Table S2 – Benchmark Comparison (literature + internal)",
        "rows": rows,
        "literature_path": os.path.abspath(literature_json),
    }


def build_all_paper_tables(
    *,
    benchmark_json: str = "",
    predictions_json: str = "",
    judge_json: str = "",
    faithfulness_json: str = "",
    literature_json: str = "",
) -> Dict[str, Any]:
    """Assemble all available tables; skip sections when input files are missing."""
    out: Dict[str, Any] = {"tables": {}}
    if predictions_json and os.path.isfile(predictions_json):
        out["tables"]["table_x_retrieval"] = table_x_retrieval(predictions_json)
    if benchmark_json and os.path.isfile(benchmark_json):
        out["tables"]["table_y_answer_quality"] = table_y_answer_quality(
            benchmark_json,
            predictions_json=predictions_json,
        )
        lit = literature_json or os.path.join(os.path.dirname(__file__), "medical_llm_literature.json")
        if os.path.isfile(lit):
            out["tables"]["table_s2_literature"] = table_s2_literature_comparison(lit, benchmark_json)
        out["tables"]["table_s1_bertscore"] = table_s1_internal_bertscore(
            benchmark_json, predictions_json=predictions_json
        )
    if judge_json and os.path.isfile(judge_json):
        out["tables"]["table_z_llm_judge"] = table_z_llm_judge(judge_json)
    if faithfulness_json and os.path.isfile(faithfulness_json):
        out["tables"]["table_h_faithfulness"] = table_h_faithfulness(faithfulness_json)
    return out


def write_paper_tables(path: str, tables: Dict[str, Any]) -> None:
    od = os.path.dirname(os.path.abspath(path))
    if od:
        os.makedirs(od, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"Wrote paper tables → {path}", flush=True)


def print_tables_summary(tables: Dict[str, Any]) -> None:
    for key, block in (tables.get("tables") or {}).items():
        title = (block or {}).get("title", key)
        rows = (block or {}).get("rows") or (block or {}).get("rows_by_configuration") or []
        print(f"\n=== {title} ({len(rows)} rows) ===", flush=True)
        print(json.dumps(rows[:8], indent=2), flush=True)
        if len(rows) > 8:
            print(f"  ... and {len(rows) - 8} more", flush=True)
