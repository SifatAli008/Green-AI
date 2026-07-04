#!/usr/bin/env python3
"""
Build a multi-chunk PubMedQA corpus for RAG (paragraph-level, not one vector per question).

Output JSONL lines:
  {"source": "pubmedqa_{pubid}_{chunk_index}", "text": "..."}

Then build FAISS:
  python build_rag_index.py --input chunks.jsonl --out_dir /kaggle/working/rag_index --add_chunk_metadata

WARNING (RAG repair plan Step 01): PubMedQA *evaluation* PMIDs must not appear in the RAG corpus.
Use train split only for corpus building; for production use PMC/guidelines sources instead.
Run ``eval_benchmarks.py --build_fixed_chunks`` on long abstracts before indexing.

Retrieval metrics treat any source matching ``pubmedqa_{pubid}`` or ``pubmedqa_{pubid}_*`` as relevant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _flatten_context(context_field: Any) -> List[str]:
    if context_field is None:
        return []
    if isinstance(context_field, dict):
        ctxs = context_field.get("contexts")
        if isinstance(ctxs, list):
            return [str(c).strip() for c in ctxs if str(c).strip()]
    raw = str(context_field).strip()
    if not raw:
        return []
    if raw.startswith("{") and "contexts" in raw:
        try:
            import ast

            d = ast.literal_eval(raw)
            ctxs = d.get("contexts")
            if isinstance(ctxs, list):
                return [str(c).strip() for c in ctxs if str(c).strip()]
        except (SyntaxError, ValueError, TypeError):
            pass
    return [raw]


def build_chunks(
    split: str = "train",
    min_chars: int = 40,
    max_items: int = 0,
    seed: int = 42,
) -> List[Dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    rows = list(ds)
    if max_items > 0 and len(rows) > max_items:
        import random

        rng = random.Random(seed)
        rows = rng.sample(rows, max_items)

    chunks: List[Dict[str, str]] = []
    for row in rows:
        pid = str(row.get("pubid", row.get("id", ""))).strip()
        if not pid:
            continue
        paras = _flatten_context(row.get("context"))
        if not paras:
            long_a = str(row.get("long_answer") or "").strip()
            if len(long_a) >= min_chars:
                paras = [long_a]
        for i, para in enumerate(paras):
            para = para.strip()
            if len(para) < min_chars:
                continue
            chunks.append({"source": f"pubmedqa_{pid}_{i}", "text": para})
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(description="Export PubMedQA paragraph chunks for RAG")
    p.add_argument("--out", default="chunks.jsonl", help="Output JSONL path")
    p.add_argument("--split", default="train", help="HF split (pqa_labeled: train)")
    p.add_argument("--min_chars", type=int, default=40, help="Skip paragraphs shorter than this")
    p.add_argument("--max_items", type=int, default=0, help="Sample N questions (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    chunks = build_chunks(
        split=args.split,
        min_chars=args.min_chars,
        max_items=args.max_items,
        seed=args.seed,
    )
    if not chunks:
        print("No chunks produced.", file=sys.stderr)
        sys.exit(2)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_pub = len({c["source"].rsplit("_", 1)[0] for c in chunks})
    print(f"Wrote {len(chunks)} chunks from ~{n_pub} PubMedQA questions -> {out_path}", flush=True)
    if len(chunks) < 2500:
        print(
            "WARNING: < 2,500 chunks — fine for smoke tests, not for publication-quality RAG.",
            flush=True,
        )


if __name__ == "__main__":
    main()
