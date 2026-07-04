#!/usr/bin/env python3
"""
Step A: build paper-ready JSON artifacts from existing CSV/JSON outputs.

Creates:
  result/benchmark_results_all.json
  result/benchmark_results_all_predictions_combined.json
  result/benchmark_results_all_predictions_combined_judge.json
  result/benchmark_results_all_predictions_combined_faithfulness.json
  result/paper_tables.json

Usage:
  python build_paper_artifacts.py
  python build_paper_artifacts.py --skip-predictions-json   # if JSON already exists
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
FAITH_FINAL = (
    ROOT
    / "fathfullness"
    / "rerun"
    / "benchmark_results_all_predictions_combined_faithfulness_final.csv"
)
JUDGE_CSV = ROOT / "LLM_Judge" / "benchmark_results_all_predictions_combined_judge.csv"
PRED_CSV = RESULT / "benchmark_results_all_predictions_combined.csv"

OUT_BENCH = RESULT / "benchmark_results_all.json"
OUT_PRED = RESULT / "benchmark_results_all_predictions_combined.json"
OUT_JUDGE = RESULT / "benchmark_results_all_predictions_combined_judge.json"
OUT_FAITH = RESULT / "benchmark_results_all_predictions_combined_faithfulness.json"
OUT_TABLES = RESULT / "paper_tables.json"

_NUMERIC_ROW_FIELDS = (
    "context_chars",
    "token_f1",
    "token_f1_label",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "gold_chunk_rank",
    "retrieval_latency_ms",
    "context_char_length",
    "evidence_token_overlap",
    "global_row_index",
)
_BOOL_ROW_FIELDS = (
    "rag_flag",
    "rag_context_used",
    "rag_context_rejected",
    "label_correct",
    "mcq_correct",
    "gold_chunk_found",
    "retrieval_evaluable",
)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("1", "true", "yes", "y")


def _maybe_float(v: Any) -> Any:
    if v is None or v == "":
        return v
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if not s:
        return ""
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s else f
    except ValueError:
        return v


def _normalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for k in _BOOL_ROW_FIELDS:
        if k in out:
            out[k] = _as_bool(out[k])
    for k in _NUMERIC_ROW_FIELDS:
        if k in out:
            out[k] = _maybe_float(out[k])
    for k in ("rag_hits", "rag_ranked_sources", "retrieved_chunk_ids", "similarity_scores", "reranker_scores", "choices_json"):
        raw = out.get(k)
        if isinstance(raw, str) and raw.strip().startswith(("[", "{")):
            try:
                out[k] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return out


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _find_benchmark_jsons() -> List[Path]:
    paths: List[Path] = []
    for folder in sorted(RESULT.iterdir()):
        if not folder.is_dir():
            continue
        for j in sorted(folder.glob("benchmark_results_all *.json"), key=lambda p: int(p.stem.split()[-1])):
            paths.append(j)
    return paths


def _weighted_mean(pairs: List[Tuple[float, int]]) -> Optional[float]:
    clean = [(m, n) for m, n in pairs if n > 0 and m == m]
    if not clean:
        return None
    total_n = sum(n for _, n in clean)
    return sum(m * n for m, n in clean) / total_n


def merge_benchmark_jsons(json_paths: List[Path]) -> Dict[str, Any]:
    """Weighted-average per-run benchmark_results_all N.json summaries."""
    metric_keys = (
        "mean",
        "std",
        "ci_lower",
        "ci_upper",
        "f1_mean",
        "f1_std",
        "f1_ci_lower",
        "f1_ci_upper",
        "bertscore_f1_mean",
        "bertscore_f1_ci_lower",
        "bertscore_f1_ci_upper",
        "rougeL_mean",
        "rougeL_ci_lower",
        "rougeL_ci_upper",
        "bleu_mean",
        "meteor_mean",
        "pubmedqa_label_accuracy_mean",
        "pubmedqa_label_accuracy_std",
        "pubmedqa_label_accuracy_ci_lower",
        "pubmedqa_label_accuracy_ci_upper",
        "mcq_accuracy_std",
    )
    accum: Dict[Tuple[str, str, str], List[Tuple[float, int]]] = defaultdict(list)
    cfg_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    bench_meta: Dict[str, Dict[str, Any]] = {}
    retrieval_health_vals: Dict[str, List[float]] = defaultdict(list)
    source_runs = 0

    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        source_runs += 1
        rh = data.get("retrieval_health") or {}
        for k in ("recall_at_1_mean", "recall_at_3_mean", "recall_at_5_mean", "recall_at_10_mean", "mrr_mean"):
            v = rh.get(k)
            if isinstance(v, (int, float)) and v == v:
                retrieval_health_vals[k].append(float(v))
        for bench_name, bench in (data.get("benchmarks") or {}).items():
            if not isinstance(bench, dict):
                continue
            bench_meta.setdefault(bench_name, {})
            for mk in ("mmlu_split", "pubmedqa_eval_split", "mmlu_pretraining_contamination_risk"):
                if bench.get(mk) and mk not in bench_meta[bench_name]:
                    bench_meta[bench_name][mk] = bench[mk]
            results = bench.get("results") or {}
            for cfg, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                n = int(metrics.get("n") or 0)
                cfg_meta[(bench_name, cfg)] = {"metric": metrics.get("metric"), "n_total_weighted": 0}
                for key in metric_keys:
                    val = metrics.get(key)
                    if isinstance(val, (int, float)) and val == val and n > 0:
                        accum[(bench_name, cfg, key)].append((float(val), n))

    merged_benches: Dict[str, Any] = {}
    for bench_name in sorted({b for b, _, _ in accum} | set(bench_meta)):
        merged_benches[bench_name] = dict(bench_meta.get(bench_name, {}))
        merged_results: Dict[str, Any] = {}
        cfgs = sorted({c for b, c, _ in accum if b == bench_name})
        for cfg in cfgs:
            block: Dict[str, Any] = dict(cfg_meta.get((bench_name, cfg), {}))
            n_pairs = accum.get((bench_name, cfg, "mean")) or accum.get((bench_name, cfg, "f1_mean")) or []
            block["n"] = sum(n for _, n in n_pairs) if n_pairs else None
            for key in metric_keys:
                wm = _weighted_mean(accum.get((bench_name, cfg, key), []))
                if wm is not None:
                    block[key] = wm
            merged_results[cfg] = block
        merged_benches[bench_name]["results"] = merged_results

    rh_out: Dict[str, Any] = {}
    for k, vals in retrieval_health_vals.items():
        if vals:
            rh_out[k] = statistics.mean(vals)
    if rh_out:
        rh_out["n_runs_averaged"] = len(json_paths)
        rh_out["healthy"] = True

    return {
        "merged_from_runs": source_runs,
        "merge_method": "weighted_mean_by_n_items",
        "benchmarks": merged_benches,
        "retrieval_health": rh_out,
    }


def export_predictions_json(csv_path: Path, out_path: Path) -> int:
    rows = [_normalize_prediction_row(r) for r in _load_csv_rows(csv_path)]
    payload = {
        "meta": {
            "source_csv": str(csv_path),
            "n_rows": len(rows),
            "export": "build_paper_artifacts.py",
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return len(rows)


def export_faithfulness_json(csv_path: Path, out_path: Path) -> Dict[str, Any]:
    from run_faithfulness_eval import aggregate_by_configuration, summarize

    rows_in = _load_csv_rows(csv_path)
    results: List[Dict[str, Any]] = []
    scores: List[float] = []
    for r in rows_in:
        raw = r.get("faithfulness_primary", "")
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if score != score:
            continue
        scores.append(score)
        results.append(
            {
                "model_name": r.get("model_name"),
                "faithfulness_primary": score,
                "hallucination_rate_primary": 100.0 - score,
                "faithfulness_model": r.get("faithfulness_model"),
                "faithfulness_context_source": r.get("faithfulness_context_source"),
                "run_folder": r.get("run_folder"),
                "benchmark": r.get("benchmark"),
                "question_id": r.get("question_id"),
            }
        )
    model = (results[0].get("faithfulness_model") if results else "") or "Qwen/Qwen2.5-7B-Instruct"
    payload = {
        "meta": {
            "source_csv": str(csv_path),
            "backend": "local_hf",
            "model_primary": model,
            "n_rows": len(results),
            "export": "build_paper_artifacts.py",
        },
        "summary_primary": summarize(scores),
        "aggregate_by_configuration": aggregate_by_configuration(results, "local_hf", model),
        "rows": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def export_judge_json(csv_path: Path, out_path: Path) -> Dict[str, Any]:
    from llm_judge import _SCORE_KEYS, _aggregate

    rows_out: List[Dict[str, Any]] = []
    parse_failures = 0
    for r in _load_csv_rows(csv_path):
        row = dict(r)
        if _as_bool(r.get("judge_parse_ok")) and str(r.get("judge_correctness", "")).strip():
            try:
                row["judge_scores"] = {
                    "correctness": float(r["judge_correctness"]),
                    "completeness": float(r["judge_completeness"]),
                    "clinical_relevance": float(r["judge_clinical_relevance"]),
                    "brief_rationale": str(r.get("judge_brief_rationale") or ""),
                }
            except (TypeError, ValueError):
                parse_failures += 1
                continue
        else:
            parse_failures += 1
            continue
        rows_out.append(row)

    judge_model = ""
    for r in rows_out:
        if r.get("judge_model"):
            judge_model = str(r["judge_model"])
            break
    if not judge_model or judge_model in (
        "meta-llama/Llama-2-7b-chat-hf:fastest",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-3B-Instruct",
    ):
        judge_model = "Qwen/Qwen2.5-7B-Instruct"

    payload = {
        "meta": {
            "source_csv": str(csv_path),
            "n_judged": len(rows_out),
            "parse_failures": parse_failures,
            "judge_model": judge_model,
            "export": "build_paper_artifacts.py",
        },
        "aggregate_by_model": _aggregate(rows_out),
        "rows": rows_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def build_paper_tables(
    *,
    benchmark_json: Path,
    predictions_json: Path,
    judge_json: Path,
    faithfulness_json: Path,
    out_tables: Path,
) -> Dict[str, Any]:
    from eval_posthoc import build_all_paper_tables, print_tables_summary, write_paper_tables

    lit = ROOT / "medical_llm_literature.json"
    tables = build_all_paper_tables(
        benchmark_json=str(benchmark_json),
        predictions_json=str(predictions_json),
        judge_json=str(judge_json),
        faithfulness_json=str(faithfulness_json),
        literature_json=str(lit),
    )
    write_paper_tables(str(out_tables), tables)
    print_tables_summary(tables)
    return tables


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Build paper JSON artifacts (Step A)")
    p.add_argument("--skip-predictions-json", action="store_true")
    p.add_argument("--skip-tables", action="store_true")
    args = p.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for req in (PRED_CSV, FAITH_FINAL, JUDGE_CSV):
        if not req.is_file():
            raise SystemExit(f"Missing required input: {req}")

    json_paths = _find_benchmark_jsons()
    if not json_paths:
        raise SystemExit(f"No benchmark_results_all *.json under {RESULT}")
    print(f"Merging {len(json_paths)} benchmark summary JSONs...", flush=True)
    bench = merge_benchmark_jsons(json_paths)
    OUT_BENCH.write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_BENCH} ({bench['merged_from_runs']} runs)", flush=True)

    if not args.skip_predictions_json or not OUT_PRED.is_file():
        print(f"Exporting predictions JSON ({PRED_CSV.name})...", flush=True)
        n = export_predictions_json(PRED_CSV, OUT_PRED)
        print(f"Wrote {OUT_PRED} ({n} rows)", flush=True)
    else:
        print(f"Skip predictions JSON — using {OUT_PRED}", flush=True)

    print("Exporting faithfulness JSON...", flush=True)
    faith = export_faithfulness_json(FAITH_FINAL, OUT_FAITH)
    print(
        f"Wrote {OUT_FAITH} (n={faith['meta']['n_rows']}, "
        f"mean={faith['summary_primary'].get('mean'):.2f})",
        flush=True,
    )

    print("Exporting judge JSON...", flush=True)
    judge = export_judge_json(JUDGE_CSV, OUT_JUDGE)
    print(
        f"Wrote {OUT_JUDGE} (n={judge['meta']['n_judged']}, "
        f"parse_failures={judge['meta']['parse_failures']})",
        flush=True,
    )

    if args.skip_tables:
        return

    print("Building paper_tables.json...", flush=True)
    build_paper_tables(
        benchmark_json=OUT_BENCH,
        predictions_json=OUT_PRED,
        judge_json=OUT_JUDGE,
        faithfulness_json=OUT_FAITH,
        out_tables=OUT_TABLES,
    )
    print(f"Wrote {OUT_TABLES}", flush=True)


if __name__ == "__main__":
    main()
