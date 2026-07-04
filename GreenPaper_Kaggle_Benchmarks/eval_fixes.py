#!/usr/bin/env python3
"""
Backward-compatible shim — all fixes are built into eval_benchmarks.py.

Prefer:
  python eval_benchmarks.py --build_fixed_chunks kaggle_working/rag_index/chunks.jsonl
  !python eval_benchmarks.py --benchmark all --rag_index_dir /kaggle/working/rag_index
"""
from __future__ import annotations

import sys

from eval_benchmarks import build_fixed_chunks, main

apply_patches = lambda: None  # noqa: E731 — no-op; eval_benchmarks is already patched


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "chunks.jsonl"
    import os

    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(inp)), "chunks_fixed.jsonl"
    )
    build_fixed_chunks(inp, out)
