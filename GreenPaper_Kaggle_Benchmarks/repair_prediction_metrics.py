#!/usr/bin/env python3
"""
Recompute token_f1 / reference_text on an existing predictions CSV or JSON
without re-running GPU inference.

Fixes degenerate 0/1 token_f1 when PubMedQA was scored vs yes/no/maybe only:
  - PubMedQA: token_f1 vs long_answer (from HF); token_f1_label vs parsed label
  - MCQ: token_f1 vs full choice text (from choices_json column)

Usage:
  python repair_prediction_metrics.py \\
      --predictions "benchmark_results_all_predictions (1).csv" \\
      --out "benchmark_results_all_predictions_repaired.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval_benchmarks import (  # noqa: E402
    KAGGLE_GOLD_RAG_DATASET_DIR,
    _load_pubmedqa,
    _mcq_token_f1,
    _parse_pubmed_model_answer,
    _prefetch_rag_for_pubmedqa_retrieval_eval,
    _pubmedqa_token_f1,
    _ranked_sources_for_metrics,
    _rag_row_diagnostic_fields,
    _retrieval_metrics_fields,
)


def _load_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("rows") or data)
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_json(path: str, rows: List[Dict[str, Any]], meta: Optional[Dict[str, Any]]) -> None:
    payload = {"meta": meta or {"repaired": True}, "n_rows": len(rows), "rows": rows}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pubmed_long_answers() -> Dict[str, str]:
    return {str(it["id"]): str(it.get("long_answer") or "").strip() for it in _load_pubmedqa()}


def _parse_ranked_sources(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x)]
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return [str(x) for x in v if str(x)]
        except json.JSONDecodeError:
            pass
    return []


def _pubmedqa_items_by_id() -> Dict[str, Dict[str, Any]]:
    return {str(it["id"]): it for it in _load_pubmedqa()}


def _recompute_retrieval_row(
    row: Dict[str, Any],
    *,
    gold_dir: str = "",
    refetch_gold: bool = False,
    items_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if str(row.get("benchmark") or "") != "pubmedqa":
        return
    rag = row.get("rag_flag")
    if rag not in (True, "True", "true", "1", 1):
        return
    qid = str(row.get("question_id") or "").strip()
    if not qid:
        return
    item = (items_by_id or {}).get(qid) or {"id": qid}
    if gold_dir:
        os.environ["GP_RAG_GOLD_INDEX_DIR"] = os.path.abspath(gold_dir)
    if refetch_gold and gold_dir:
        _block, met_hits, met_ranked, _src, met_diag = _prefetch_rag_for_pubmedqa_retrieval_eval(item)
        ranked = _ranked_sources_for_metrics(met_ranked, met_hits, met_diag)
        row["rag_ranked_sources"] = json.dumps(ranked, ensure_ascii=False)
        m = _retrieval_metrics_fields("pubmedqa", item, True, ranked, met_hits, met_diag)
        row.update(_rag_row_diagnostic_fields("pubmedqa", item, True, ranked, met_hits, met_diag))
        for k, v in m.items():
            row[k] = v
        return
    ranked = _parse_ranked_sources(row.get("rag_ranked_sources"))
    diag: Dict[str, Any] = {}
    if gold_dir:
        diag["retrieval_eval_index_dir"] = os.path.abspath(gold_dir)
    for key in ("retrieved_chunk_ids",):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                diag[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    ranked = _ranked_sources_for_metrics(ranked, [], diag)
    m = _retrieval_metrics_fields("pubmedqa", item, True, ranked, [], diag)
    for k, v in m.items():
        row[k] = v


def repair_rows(
    rows: List[Dict[str, Any]],
    *,
    recompute_retrieval: bool = True,
    gold_dir: str = "",
    refetch_gold: bool = False,
) -> List[Dict[str, Any]]:
    long_by_id = _pubmed_long_answers()
    items_by_id = _pubmedqa_items_by_id() if refetch_gold else None
    out: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        bench = str(row.get("benchmark") or "")
        if bench in ("medqa", "mmlu_med"):
            try:
                choices = json.loads(str(row.get("choices_json") or "[]"))
            except json.JSONDecodeError:
                choices = []
            gold = str(row.get("reference_answer") or "").strip().upper()
            pred = str(row.get("parsed_prediction") or row.get("model_answer") or "").strip().upper()
            f1_text, f1_label = _mcq_token_f1(pred, gold, choices)
            row["reference_text"] = row.get("reference_text") or ""
            row["token_f1"] = f1_text
            row["token_f1_label"] = f1_label
        elif bench == "pubmedqa":
            qid = str(row.get("question_id") or "").strip()
            label = str(row.get("reference_answer") or "").strip().lower()
            long_a = long_by_id.get(qid, "")
            ref_text = long_a if long_a else label
            raw = str(row.get("raw_response") or row.get("model_answer") or "")
            parsed = _parse_pubmed_model_answer(raw)
            f1_main, f1_lbl = _pubmedqa_token_f1(raw, parsed, label, ref_text)
            row["reference_text"] = ref_text
            row["token_f1"] = f1_main
            row["token_f1_label"] = f1_lbl
        if recompute_retrieval:
            _recompute_retrieval_row(
                row,
                gold_dir=gold_dir,
                refetch_gold=refetch_gold,
                items_by_id=items_by_id if refetch_gold else None,
            )
        out.append(row)
    return out


def _summarize(rows: List[Dict[str, Any]]) -> None:
    for bench in ("medqa", "mmlu_med", "pubmedqa"):
        br = [r for r in rows if r.get("benchmark") == bench]
        if not br:
            continue
        vals = [float(r["token_f1"]) for r in br if str(r.get("token_f1", "")).strip()]
        frac = [v for v in vals if 0 < v < 1]
        print(
            f"{bench}: token_f1 n={len(vals)} unique={len(set(round(v, 3) for v in vals))} "
            f"fractional={len(frac)} ({100 * len(frac) / max(1, len(vals)):.1f}%)",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Repair token_f1 on saved predictions")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True, help="Output .csv or .json path")
    p.add_argument(
        "--no_recompute_retrieval",
        action="store_true",
        help="Skip recall@k / MRR recompute from rag_ranked_sources (PubMedQA RAG rows only).",
    )
    p.add_argument(
        "--gold_rag_dir",
        default=KAGGLE_GOLD_RAG_DATASET_DIR,
        help=(
            "PubMedQA gold index dir (default on Kaggle: "
            f"{KAGGLE_GOLD_RAG_DATASET_DIR})."
        ),
    )
    p.add_argument(
        "--refetch_gold_retrieval",
        action="store_true",
        help="Re-run gold-index hybrid retrieval per PubMedQA RAG row (CPU; fixes idx_* / empty ranked).",
    )
    args = p.parse_args()

    rows = _load_rows(args.predictions)
    if not rows:
        print("No rows.", file=sys.stderr)
        sys.exit(2)

    gold_dir = (args.gold_rag_dir or "").strip()
    refetch = bool(args.refetch_gold_retrieval and gold_dir)
    repaired = repair_rows(
        rows,
        recompute_retrieval=not args.no_recompute_retrieval,
        gold_dir=gold_dir,
        refetch_gold=refetch,
    )
    _summarize(repaired)

    out = Path(args.out)
    if out.suffix.lower() == ".json":
        meta = {"source": str(args.predictions), "repaired_metrics": True}
        _write_json(str(out), repaired, meta)
    else:
        fields = list(rows[0].keys())
        for extra in ("reference_text", "token_f1_label"):
            if extra not in fields:
                fields.insert(fields.index("reference_answer") + 1 if "reference_answer" in fields else len(fields), extra)
        _write_csv(str(out), repaired, fields)

    print(f"Wrote {out} ({len(repaired)} rows)", flush=True)


if __name__ == "__main__":
    main()
