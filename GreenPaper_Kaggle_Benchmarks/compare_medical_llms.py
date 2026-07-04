#!/usr/bin/env python3
"""
Medical LLM comparison tables (faculty plan §6).

Plan A: internal BERTScore / accuracy from ``eval_benchmarks`` JSON.
Plan B: literature MedQA/PubMedQA rows from ``medical_llm_literature.json``.

  python compare_medical_llms.py --benchmark_json benchmark_results_all.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from eval_posthoc import (
    build_all_paper_tables,
    table_s1_internal_bertscore,
    table_s2_literature_comparison,
    write_paper_tables,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Medical LLM comparison tables S1/S2")
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--predictions_json", default="")
    p.add_argument(
        "--literature_json",
        default=os.path.join(os.path.dirname(__file__), "medical_llm_literature.json"),
    )
    p.add_argument("--out", default="", help="Write JSON (default: medical_llm_comparison.json beside benchmark)")
    args = p.parse_args()

    bench = os.path.abspath(args.benchmark_json)
    if not os.path.isfile(bench):
        print(f"Not found: {bench}", file=sys.stderr)
        sys.exit(2)

    lit = os.path.abspath(args.literature_json)
    pred = os.path.abspath(args.predictions_json) if args.predictions_json else ""
    out_path = args.out.strip() or os.path.join(
        os.path.dirname(bench), "medical_llm_comparison.json"
    )

    payload = {
        "table_s1": table_s1_internal_bertscore(bench, predictions_json=pred),
        "table_s2": table_s2_literature_comparison(lit, bench),
    }
    write_paper_tables(out_path, {"tables": payload})
    print(json.dumps(payload, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
