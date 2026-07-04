#!/usr/bin/env python3
"""
Build FAISS IndexFlatIP + chunks.jsonl for real_model_runner RAG.

Input: JSONL with one object per line; each object should include a \"text\" field
(the chunk body). Other keys are preserved in chunks.jsonl for traceability.

Output (default --out_dir):
  index.faiss   — L2-normalized vectors, inner-product search (= cosine)
  chunks.jsonl — same line order as row i in the index

Usage:
  pip install -q sentence-transformers faiss-cpu
  python build_rag_index.py --input corpus_chunks.jsonl --out_dir /kaggle/working/rag_index

Then run benchmarks:
  python eval_benchmarks.py --rag_index_dir /kaggle/working/rag_index ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def load_texts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                rows.append({"text": obj})
            elif isinstance(obj, dict):
                if "text" not in obj and "chunk" in obj:
                    obj = {**obj, "text": obj.get("chunk", "")}
                rows.append(obj)
            else:
                rows.append({"text": str(obj)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS index + chunks.jsonl for RAG")
    p.add_argument("--input", required=True, help="Input JSONL (objects with \"text\" field)")
    p.add_argument("--out_dir", default="rag_index", help="Directory for index.faiss + chunks.jsonl")
    p.add_argument(
        "--embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Must match RAG_EMBED_MODEL at query time. "
        "Biomedical alternative: BAAI/bge-base-en-v1.5 (768-dim; rebuild index after switch).",
    )
    p.add_argument(
        "--add_chunk_metadata",
        action="store_true",
        help="Add chunk_id and char_offset fields to each JSONL row before indexing.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    try:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("pip install sentence-transformers faiss-cpu numpy", file=sys.stderr)
        raise e

    rows = load_texts(args.input)
    texts = [str(r.get("text", "")).strip() for r in rows]
    if not any(texts):
        print("No non-empty \"text\" fields in input.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    chunk_path = os.path.join(args.out_dir, "chunks.jsonl")
    with open(chunk_path, "w", encoding="utf-8") as f:
        offset = 0
        for i, r in enumerate(rows):
            if args.add_chunk_metadata:
                body = str(r.get("text", ""))
                r = dict(r)
                r.setdefault("chunk_id", str(r.get("source") or f"chunk_{i}"))
                r.setdefault("char_offset", offset)
                offset += len(body)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    st = SentenceTransformer(args.embed_model)
    dim = st.get_sentence_embedding_dimension()
    batches: List[Any] = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        try:
            e = st.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            e = st.encode(batch, convert_to_numpy=True)
        if e.dtype != np.float32:
            e = e.astype(np.float32)
        faiss.normalize_L2(e)
        batches.append(e)

    mat = np.vstack(batches)
    try:
        from rag_retrieval import validate_embedding_matrix

        validate_embedding_matrix(mat)
        print("Embedding validation OK (no NaN/zero vectors; L2 norms ~ 1).", flush=True)
    except ImportError:
        if int(np.isnan(mat).sum()) > 0:
            raise ValueError("embeddings contain NaN")
        norms = np.linalg.norm(mat, axis=1)
        if not (norms > 0).all():
            raise ValueError("embeddings contain zero vectors")
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    if index.ntotal != len(rows):
        raise RuntimeError(f"Index count mismatch: ntotal={index.ntotal} vs chunks={len(rows)}")
    idx_path = os.path.join(args.out_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    print(f"Wrote {idx_path} (ntotal={index.ntotal}, dim={dim}, IndexFlatIP cosine)", flush=True)
    print(f"Wrote {chunk_path}", flush=True)
    print("Use: --rag_index_dir", os.path.abspath(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
