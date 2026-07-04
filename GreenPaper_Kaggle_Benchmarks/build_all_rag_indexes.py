#!/usr/bin/env python3
"""
Build or verify all RAG corpora + FAISS indexes for paper benchmarks.

Outputs (default under kaggle_working/):
  rag_index/
    chunks.jsonl + index.faiss              — ~25k external (MedQuad + PubMed)
    chunks_paper_556.jsonl + index_paper_556.faiss — 115-doc IEEE corpus
  rag_index_gold/
    chunks.jsonl + index.faiss + pubmedqa_gold_manifest.json — PubMedQA recall@k

Usage:
  python build_all_rag_indexes.py              # verify only (skip if complete)
  python build_all_rag_indexes.py --rebuild    # rebuild everything
  python build_all_rag_indexes.py --rebuild gold   # rebuild gold only

Requires: pip install sentence-transformers faiss-cpu datasets transformers
Rebuild of external/paper corpora downloads Hugging Face datasets (network).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "kaggle_working"
PRIMARY_DIR = WORKDIR / "rag_index"
GOLD_DIR = WORKDIR / "rag_index_gold"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd or ROOT))


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _verify_pair(
    label: str,
    chunks_path: Path,
    faiss_path: Path,
) -> Tuple[bool, str]:
    if not chunks_path.is_file():
        return False, f"{label}: missing {chunks_path.name}"
    if not faiss_path.is_file():
        return False, f"{label}: missing {faiss_path.name}"
    try:
        import faiss  # type: ignore
    except ImportError:
        return False, f"{label}: pip install faiss-cpu"
    n_chunks = _count_lines(chunks_path)
    idx = faiss.read_index(str(faiss_path))
    if idx.ntotal != n_chunks:
        return False, f"{label}: FAISS ntotal={idx.ntotal} != chunks={n_chunks}"
    return True, f"{label}: OK ({n_chunks} chunks, dim={idx.d})"


def verify_all() -> bool:
    checks = [
        _verify_pair("primary", PRIMARY_DIR / "chunks.jsonl", PRIMARY_DIR / "index.faiss"),
        _verify_pair(
            "paper_556",
            PRIMARY_DIR / "chunks_paper_556.jsonl",
            PRIMARY_DIR / "index_paper_556.faiss",
        ),
        _verify_pair("gold", GOLD_DIR / "chunks.jsonl", GOLD_DIR / "index.faiss"),
    ]
    ok = True
    for good, msg in checks:
        print(msg, flush=True)
        ok = ok and good

    manifest = GOLD_DIR / "pubmedqa_gold_manifest.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        print(
            f"gold manifest: eval_holdout={meta.get('n_eval_holdout')} "
            f"corpus={meta.get('n_corpus_for_index')} chunks_file={_count_lines(GOLD_DIR / 'chunks.jsonl')}",
            flush=True,
        )
    stats = PRIMARY_DIR / "paper_corpus_stats.json"
    if stats.is_file():
        meta = json.loads(stats.read_text(encoding="utf-8"))
        print(
            f"paper corpus: docs={meta.get('n_documents')} chunks={meta.get('n_chunks')}",
            flush=True,
        )
    return ok


def build_primary(*, force: bool) -> None:
    chunks = PRIMARY_DIR / "chunks.jsonl"
    faiss_p = PRIMARY_DIR / "index.faiss"
    if not force and verify_pair_quiet(chunks, faiss_p):
        print("primary: already built — skip", flush=True)
        return
    PRIMARY_DIR.mkdir(parents=True, exist_ok=True)
    external = PRIMARY_DIR / "external_chunks.jsonl"
    _run(
        [
            sys.executable,
            "build_external_corpus.py",
            "--out",
            str(external),
            "--target_chunks",
            "25000",
        ]
    )
    _run(
        [
            sys.executable,
            "build_rag_index.py",
            "--input",
            str(external),
            "--out_dir",
            str(PRIMARY_DIR),
            "--add_chunk_metadata",
            "--embed_model",
            EMBED,
        ]
    )


def build_paper(*, force: bool) -> None:
    chunks = PRIMARY_DIR / "chunks_paper_556.jsonl"
    faiss_p = PRIMARY_DIR / "index_paper_556.faiss"
    if not force and verify_pair_quiet(chunks, faiss_p):
        print("paper_556: already built — skip", flush=True)
        return
    PRIMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_chunks = PRIMARY_DIR / "chunks_paper_556.jsonl"
    _run(
        [
            sys.executable,
            "build_paper_corpus.py",
            "--out",
            str(out_chunks),
            "--manifest_out",
            str(PRIMARY_DIR / "paper_corpus_stats.json"),
        ]
    )
    # build_rag_index writes chunks.jsonl + index.faiss — use a temp dir then rename
    tmp = PRIMARY_DIR / "_paper_build_tmp"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
    else:
        tmp.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "build_rag_index.py",
            "--input",
            str(out_chunks),
            "--out_dir",
            str(tmp),
            "--add_chunk_metadata",
            "--embed_model",
            EMBED,
        ]
    )
    (tmp / "index.faiss").replace(PRIMARY_DIR / "index_paper_556.faiss")
    if (tmp / "chunks.jsonl").read_text(encoding="utf-8") != out_chunks.read_text(encoding="utf-8"):
        pass  # keep build_paper_corpus output as canonical chunks_paper_556.jsonl
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def build_gold(*, force: bool) -> None:
    chunks = GOLD_DIR / "chunks.jsonl"
    faiss_p = GOLD_DIR / "index.faiss"
    if not force and verify_pair_quiet(chunks, faiss_p):
        print("gold: already built — skip", flush=True)
        return
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "eval_benchmarks.py",
            "--build_gold_index",
            str(GOLD_DIR),
        ]
    )


def verify_pair_quiet(chunks_path: Path, faiss_path: Path) -> bool:
    ok, _ = _verify_pair("x", chunks_path, faiss_path)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Build or verify all RAG indexes")
    p.add_argument(
        "--rebuild",
        nargs="*",
        choices=["primary", "paper", "gold", "all"],
        help="Rebuild selected indexes (default: verify only)",
    )
    args = p.parse_args()
    targets = set(args.rebuild or [])

    if not targets:
        if verify_all():
            print("\nAll RAG indexes verified — nothing to rebuild.", flush=True)
            print(f"  primary: {PRIMARY_DIR}", flush=True)
            print(f"  gold:    {GOLD_DIR}", flush=True)
            return
        print("\nSome indexes missing or invalid — rebuilding all.", flush=True)
        targets = {"all"}

    do_all = "all" in targets
    if do_all or "primary" in targets:
        build_primary(force=True)
    if do_all or "paper" in targets:
        build_paper(force=True)
    if do_all or "gold" in targets:
        build_gold(force=True)

    if not verify_all():
        sys.exit(1)
    print("\nRAG preprocessing complete.", flush=True)


if __name__ == "__main__":
    main()
