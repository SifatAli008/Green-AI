#!/usr/bin/env python3
"""
Build a PubMedQA gold-aligned RAG corpus + FAISS index for eval_benchmarks.py.

Use when the default ~25k external index has no ``pubmedqa_*`` sources (recall@k = 0,
RAG often hurts task accuracy). Chunks use ``pubmedqa_<pmid>`` (first window) and
``pubmedqa_<pmid>_<n>`` for overlaps — same prefix rules as ``eval_quality_metrics``.

WARNING: This corpus is built from the same HF split used for PubMedQA *evaluation*
(``pqa_labeled`` train). Retrieval recall@k and RAG gains are **in-distribution** /
leakage-prone vs the external non-leaking 25k index. Report that clearly in the paper.

Quick start (Kaggle):
  python build_pubmedqa_gold_corpus.py --out_dir /kaggle/working/rag_index_gold
  python eval_benchmarks.py --benchmark all --max_items 500 --seed 42 \\
      --rag_index_dir /kaggle/working/rag_index_gold --rag_top_k 10 \\
      --out_json /kaggle/working/benchmark_results_gold.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def flatten_context(ctx: Any) -> str:
    if not ctx:
        return ""
    s = str(ctx).strip()
    if s.startswith("{") and "contexts" in s:
        try:
            import ast

            d = ast.literal_eval(s)
            parts = d.get("contexts") or []
            if parts:
                return "\n\n".join(str(p).strip() for p in parts if p)
        except (SyntaxError, ValueError, TypeError):
            pass
    return s


def build_gold_chunks(
    *,
    split: str = "train",
    chunk_words: int = 120,
    overlap_words: int = 24,
    min_subchunk_words: int = 20,
) -> List[Dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    chunks: List[Dict[str, str]] = []
    for row in ds:
        pid = str(row.get("pubid") or row.get("id") or "").strip()
        ctx = flatten_context(row.get("context"))
        text = ctx or str(row.get("long_answer") or "")
        if not text.strip() or not pid:
            continue
        words = text.split()
        source_base = f"pubmedqa_{pid}"
        if len(words) <= chunk_words:
            chunks.append({"text": text, "source": source_base})
        else:
            step = max(1, chunk_words - overlap_words)
            i, idx = 0, 0
            while i < len(words):
                window = words[i : i + chunk_words]
                if len(window) < min_subchunk_words:
                    break
                sub_src = source_base if idx == 0 else f"{source_base}_{idx}"
                chunks.append({"text": " ".join(window), "source": sub_src})
                idx += 1
                i += step
    return chunks


def write_chunks(chunks: List[Dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def verify_chunks(chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    exact = {c["source"] for c in chunks if re.match(r"^pubmedqa_\d+$", c["source"])}
    bases: Counter[str] = Counter()
    for c in chunks:
        m = re.match(r"^pubmedqa_\d+", c["source"])
        if m:
            bases[m.group()] += 1
    per_abs = Counter(bases.values())
    return {
        "n_chunks": len(chunks),
        "n_exact_gold_sources": len(exact),
        "chunks_per_abstract": dict(sorted(per_abs.items())),
    }


def build_faiss_index(
    chunks: List[Dict[str, str]],
    out_dir: str,
    embed_model: str = DEFAULT_EMBED,
    batch_size: int = 128,
) -> Tuple[str, str, List[str]]:
    import faiss  # type: ignore
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    faiss_path = os.path.join(out_dir, "index.faiss")
    write_chunks(chunks, chunks_path)
    faiss.write_index(index, faiss_path)
    return chunks_path, faiss_path, sources


def _doc_matches(doc_id: str, gold: str) -> bool:
    doc_id, gold = str(doc_id), str(gold)
    return doc_id == gold or doc_id.startswith(gold + "_")


def _recall_at_k(ranked: List[str], gold: str, k: int) -> float:
    top = ranked[: max(1, k)]
    return 1.0 if any(_doc_matches(d, gold) for d in top) else 0.0


def _mrr(ranked: List[str], gold: str) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if _doc_matches(doc_id, gold):
            return 1.0 / float(rank)
    return 0.0


def recall_health_check(
    chunks: List[Dict[str, str]],
    index_path: str,
    embed_model: str = DEFAULT_EMBED,
    n_test: int = 50,
    split: str = "train",
) -> Dict[str, float]:
    """Prefix-match recall@k (same rule as eval_quality_metrics)."""
    import faiss  # type: ignore
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    sources = [c["source"] for c in chunks]
    index = faiss.read_index(index_path)
    model = SentenceTransformer(embed_model)
    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)

    buckets: Dict[str, List[float]] = {
        "recall_at_1": [],
        "recall_at_3": [],
        "recall_at_5": [],
        "recall_at_10": [],
        "mrr": [],
    }
    for i, row in enumerate(ds):
        if i >= n_test:
            break
        pid = str(row.get("pubid") or row.get("id") or "").strip()
        q = str(row.get("question") or "").strip()
        gold = f"pubmedqa_{pid}"
        if not q or not pid:
            continue
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _d, idxs = index.search(qv, min(10, index.ntotal))
        ranked = [sources[j] for j in idxs[0]]
        buckets["recall_at_1"].append(_recall_at_k(ranked, gold, 1))
        buckets["recall_at_3"].append(_recall_at_k(ranked, gold, 3))
        buckets["recall_at_5"].append(_recall_at_k(ranked, gold, 5))
        buckets["recall_at_10"].append(_recall_at_k(ranked, gold, 10))
        buckets["mrr"].append(_mrr(ranked, gold))

    if not buckets["recall_at_1"]:
        return {}
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Build PubMedQA gold-aligned RAG index")
    p.add_argument("--out_dir", default="/kaggle/working/rag_index_gold")
    p.add_argument("--split", default="train", help="HF split for pqa_labeled")
    p.add_argument("--chunk_words", type=int, default=120)
    p.add_argument("--overlap_words", type=int, default=24)
    p.add_argument("--embed_model", default=DEFAULT_EMBED)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--skip_faiss", action="store_true", help="Only write chunks.jsonl")
    p.add_argument("--recall_check", type=int, default=50, help="N queries for sanity (0=skip)")
    args = p.parse_args()

    print(f"Loading PubMedQA pqa_labeled ({args.split})...", flush=True)
    chunks = build_gold_chunks(
        split=args.split,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    if not chunks:
        print("No chunks produced.", file=sys.stderr)
        sys.exit(2)

    stats = verify_chunks(chunks)
    print(json.dumps(stats, indent=2), flush=True)
    if stats["n_exact_gold_sources"] < 900:
        print(
            f"WARNING: only {stats['n_exact_gold_sources']} exact pubmedqa_<id> sources (expected ~1000).",
            flush=True,
        )

    out_dir = os.path.abspath(args.out_dir)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    if args.skip_faiss:
        write_chunks(chunks, chunks_path)
        print(f"Wrote {chunks_path} ({len(chunks)} chunks)", flush=True)
        return

    chunks_path, faiss_path, sources = build_faiss_index(
        chunks, out_dir, embed_model=args.embed_model, batch_size=args.batch_size
    )
    print(f"Wrote {chunks_path} and {faiss_path}", flush=True)

    import faiss  # type: ignore
    import numpy as np
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(faiss_path)
    model = SentenceTransformer(args.embed_model)
    q_emb = model.encode(
        [chunks[0]["text"]],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    _d, idxs = index.search(q_emb, 3)
    rank0 = sources[int(idxs[0][0])]
    print(f"Sanity rank-1 for chunk[0]: {rank0!r} (expected {sources[0]!r})", flush=True)

    if args.recall_check > 0:
        metrics = recall_health_check(
            chunks, faiss_path, embed_model=args.embed_model, n_test=args.recall_check
        )
        print("Retrieval health (prefix match, eval_quality_metrics):", flush=True)
        for k, v in metrics.items():
            print(f"  {k} = {v:.3f}", flush=True)
        r1, r3 = metrics.get("recall_at_1", 0), metrics.get("recall_at_3", 0)
        if r3 <= r1:
            print(
                f"WARNING: recall@3 ({r3:.3f}) should exceed recall@1 ({r1:.3f}) "
                "when sub-chunks spread gold rank.",
                flush=True,
            )


if __name__ == "__main__":
    main()
