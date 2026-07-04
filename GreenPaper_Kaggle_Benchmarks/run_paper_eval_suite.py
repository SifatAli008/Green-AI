#!/usr/bin/env python3
"""
Orchestrate faculty-review post-hoc evaluation (plan in ``current plan on faculty reviwe.md``).

Typical workflow after ``eval_benchmarks.py``:

  python eval_benchmarks.py --benchmark all --max_items 500 --seed 42
  python run_paper_eval_suite.py \\
      --benchmark_json benchmark_results_all.json \\
      --predictions_json benchmark_results_all_predictions.json \\
      --run_judge --run_faithfulness \\
      --faithfulness_limit 200

Or one shot from benchmarks:

  python eval_benchmarks.py --benchmark all --max_items 500 --post_eval

Outputs:
  - ``<predictions_stem>_judge.json`` (Table 3)
  - ``<predictions_stem>_faithfulness.json`` (Table 4)
  - ``paper_tables.json`` (Tables X, Y, Z, H, S1, S2)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional


def _stem(path: str) -> str:
    p = os.path.abspath(path)
    if p.endswith(".json"):
        return p[:-5]
    return p


def run_post_eval(
    *,
    benchmark_json: str,
    predictions_json: str,
    out_tables: str = "",
    run_judge: bool = False,
    run_faithfulness: bool = False,
    judge_max_rows: int = 0,
    judge_seed: Optional[int] = None,
    judge_hf_router: bool = False,
    judge_model: str = "",
    faithfulness_limit: int = 0,
    faithfulness_backend: str = "auto",
    faithfulness_out: str = "",
    skip_tables: bool = False,
) -> Dict[str, Any]:
    """Run optional judge + faithfulness, then build paper tables."""
    pred_abs = os.path.abspath(predictions_json)
    bench_abs = os.path.abspath(benchmark_json)
    if not os.path.isfile(pred_abs):
        raise FileNotFoundError(f"Predictions not found: {pred_abs}")
    if not os.path.isfile(bench_abs):
        raise FileNotFoundError(f"Benchmark JSON not found: {bench_abs}")

    pred_stem = _stem(pred_abs)
    judge_path = f"{pred_stem}_judge.json"
    faith_path = faithfulness_out.strip() or f"{pred_stem}_faithfulness.json"
    tables_path = out_tables.strip() or os.path.join(
        os.path.dirname(bench_abs) or ".", "paper_tables.json"
    )

    if run_judge:
        from llm_judge import judge_predictions

        print("\n=== LLM-as-judge (Table 3) ===", flush=True)
        judge_predictions(
            pred_abs,
            out=judge_path,
            model=judge_model,
            hf_router=judge_hf_router,
            max_rows=judge_max_rows,
            seed=judge_seed,
        )
    elif os.path.isfile(judge_path):
        print(f"Using existing judge output: {judge_path}", flush=True)
    else:
        judge_path = ""

    if run_faithfulness:
        from run_faithfulness_eval import load_predictions_json, run_faithfulness_eval

        print("\n=== Faithfulness / hallucination (Table 4) ===", flush=True)
        rows = load_predictions_json(pred_abs)
        if faithfulness_limit > 0:
            rows = rows[: faithfulness_limit]
        if not rows:
            print(
                "WARNING: no RAG rows with context+answer for faithfulness "
                "(need retrieved_context or PubMedQA context).",
                flush=True,
            )
        else:
            run_faithfulness_eval(
                rows,
                out_path=faith_path,
                backend=faithfulness_backend,
            )
    elif os.path.isfile(faith_path):
        print(f"Using existing faithfulness output: {faith_path}", flush=True)
    else:
        faith_path = ""

    tables: Dict[str, Any] = {}
    if not skip_tables:
        from eval_posthoc import build_all_paper_tables, print_tables_summary, write_paper_tables

        lit = os.path.join(os.path.dirname(__file__), "medical_llm_literature.json")
        print("\n=== Building paper tables ===", flush=True)
        tables = build_all_paper_tables(
            benchmark_json=bench_abs,
            predictions_json=pred_abs,
            judge_json=judge_path,
            faithfulness_json=faith_path,
            literature_json=lit,
        )
        write_paper_tables(tables_path, tables)
        print_tables_summary(tables)

    return {
        "benchmark_json": bench_abs,
        "predictions_json": pred_abs,
        "judge_json": judge_path or None,
        "faithfulness_json": faith_path or None,
        "paper_tables_json": tables_path if not skip_tables else None,
        "tables": tables.get("tables") if tables else {},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Faculty-review post-hoc eval suite")
    p.add_argument("--benchmark_json", required=True, help="benchmark_results_all.json from eval_benchmarks")
    p.add_argument("--predictions_json", required=True, help="*_predictions.json from eval_benchmarks")
    p.add_argument("--out_tables", default="", help="Output paper_tables.json path")
    p.add_argument("--run_judge", action="store_true", help="Run llm_judge.py (Table 3)")
    p.add_argument("--run_faithfulness", action="store_true", help="Run faithfulness scoring (Table 4)")
    p.add_argument("--judge_max_rows", type=int, default=0, help="Cap judge rows (0=all)")
    p.add_argument("--judge_seed", type=int, default=None)
    p.add_argument("--judge_hf_router", action="store_true")
    p.add_argument("--judge_model", default="", help="GP_JUDGE_MODEL override")
    p.add_argument("--faithfulness_limit", type=int, default=0, help="Cap faithfulness rows (0=all RAG rows)")
    p.add_argument(
        "--faithfulness_backend",
        default="auto",
        choices=["auto", "openrouter", "deepeval"],
    )
    p.add_argument("--faithfulness_out", default="", help="Faithfulness JSON output path")
    p.add_argument("--skip_tables", action="store_true", help="Only run judge/faithfulness, no paper_tables.json")
    args = p.parse_args()

    if not args.run_judge and not args.run_faithfulness and not args.skip_tables:
        print(
            "Nothing to run: pass --run_judge and/or --run_faithfulness, "
            "or omit --skip_tables to only rebuild tables from existing artifacts.",
            flush=True,
        )

    try:
        run_post_eval(
            benchmark_json=args.benchmark_json,
            predictions_json=args.predictions_json,
            out_tables=args.out_tables,
            run_judge=args.run_judge,
            run_faithfulness=args.run_faithfulness,
            judge_max_rows=args.judge_max_rows,
            judge_seed=args.judge_seed,
            judge_hf_router=args.judge_hf_router,
            judge_model=args.judge_model,
            faithfulness_limit=args.faithfulness_limit,
            faithfulness_backend=args.faithfulness_backend,
            faithfulness_out=args.faithfulness_out,
            skip_tables=args.skip_tables,
        )
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
