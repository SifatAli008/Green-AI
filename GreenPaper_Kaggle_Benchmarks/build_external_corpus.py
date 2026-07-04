#!/usr/bin/env python3
"""
Build a large non-leaking biomedical RAG corpus (RAG Repair Plan Step 01).

Sources (no PubMedQA / MedQA / MMLU-med benchmark text):
  1. NIH MedQuad consumer-health answers (keivalya/MedQuad-MedicalQnADataset)
  2. PubMed abstracts (slinusc/PubMedAbstractsSubset), PMID-filtered

Exclusions:
  - All PubMedQA ``pubid`` values (pqa_labeled, pqa_artificial, pqa_unlabeled)
  - PMIDs from an existing ``pubmedqa_*`` chunks.jsonl (legacy corpus)
  - Optional: benchmark context prefix fingerprints (PubMedQA contexts)

Output JSONL per chunk:
  {source, title, section, pmid, chunk_id, char_offset, text}

Usage:
  python build_external_corpus.py --out kaggle_working/rag_index/external_chunks.jsonl --target_chunks 25000
  python build_rag_index.py --input kaggle_working/rag_index/external_chunks.jsonl --out_dir kaggle_working/rag_index --add_chunk_metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip().lower())


def _fingerprint(s: str, n: int = 160) -> str:
    return hashlib.sha256(_norm_text(s)[:2000].encode("utf-8")).hexdigest()[:n]


def chunk_text_windows(
    text: str,
    *,
    source_prefix: str,
    title: str = "",
    section: str = "",
    pmid: str = "",
    chunk_words: int = 300,
    overlap_words: int = 54,
    min_words: int = 40,
) -> List[Dict[str, Any]]:
    """~256-384 token windows with ~18% overlap and metadata."""
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_words:
        if len(words) < min_words:
            return []
        cid = f"{source_prefix}_0"
        return [
            {
                "source": source_prefix,
                "title": title,
                "section": section,
                "pmid": pmid,
                "chunk_id": cid,
                "char_offset": 0,
                "text": text,
            }
        ]
    step = max(1, chunk_words - overlap_words)
    out: List[Dict[str, Any]] = []
    i, idx, off = 0, 0, 0
    while i < len(words):
        window = words[i : i + chunk_words]
        if len(window) < min_words:
            break
        body = " ".join(window)
        sub = source_prefix if idx == 0 else f"{source_prefix}_{idx}"
        cid = f"{sub}_c{idx}"
        out.append(
            {
                "source": sub,
                "title": title,
                "section": section,
                "pmid": pmid,
                "chunk_id": cid,
                "char_offset": off,
                "text": body,
            }
        )
        off += len(body)
        idx += 1
        i += step
    return out


def load_pubmedqa_excluded_pubids() -> Set[str]:
    from datasets import load_dataset

    excluded: Set[str] = set()
    for config in ("pqa_labeled", "pqa_artificial", "pqa_unlabeled"):
        try:
            dset = load_dataset("pubmed_qa", config)
        except Exception as ex:
            print(f"  skip pubmed_qa/{config}: {ex}", flush=True)
            continue
        splits = list(dset.keys()) if hasattr(dset, "keys") else ["train"]
        for split in splits:
            for row in dset[split]:
                pid = str(row.get("pubid", row.get("id", ""))).strip()
                if pid:
                    excluded.add(pid)
    print(f"Excluded PubMedQA pubids: {len(excluded)}", flush=True)
    return excluded


def load_legacy_pubmedqa_pmids(chunks_path: str) -> Set[str]:
    pmids: Set[str] = set()
    if not os.path.isfile(chunks_path):
        return pmids
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = str(obj.get("source") or "")
            m = re.match(r"pubmedqa_(\d+)", src)
            if m:
                pmids.add(m.group(1))
    print(f"Legacy pubmedqa_* PMIDs from {chunks_path}: {len(pmids)}", flush=True)
    return pmids


def load_context_fingerprints() -> Set[str]:
    """Fingerprints of PubMedQA evaluation contexts (substring leak guard)."""
    from datasets import load_dataset

    fps: Set[str] = set()
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    except Exception:
        return fps
    for row in ds:
        ctx = row.get("context")
        if isinstance(ctx, dict):
            parts = ctx.get("contexts") or []
            blob = " ".join(str(p) for p in parts)
        else:
            blob = str(ctx or "")
        la = str(row.get("long_answer") or "")
        q = str(row.get("question") or "")
        for block in (blob, la, q):
            if len(block) > 80:
                fps.add(_fingerprint(block))
    print(f"PubMedQA context fingerprints: {len(fps)}", flush=True)
    return fps


def _leaks_benchmark(fps: Set[str], text: str) -> bool:
    if not fps or len(text) < 80:
        return False
    return _fingerprint(text) in fps


def iter_medquad_chunks(
    *,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    context_fps: Set[str],
) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    print(f"MedQuad: {len(ds)} Q&A rows", flush=True)
    for i, row in enumerate(ds):
        qtype = str(row.get("qtype") or "general")
        answer = str(row.get("Answer") or "").strip()
        if _leaks_benchmark(context_fps, answer):
            continue
        title = f"MedQuad: {qtype}"
        prefix = f"medquad_{i}"
        for ch in chunk_text_windows(
            answer,
            source_prefix=prefix,
            title=title,
            section=qtype,
            pmid="",
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
        ):
            yield ch


def iter_pubmed_abstract_chunks(
    excluded_pmids: Set[str],
    *,
    target: int,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
    context_fps: Set[str],
    max_scan: int = 500_000,
) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("slinusc/PubMedAbstractsSubset", split="train", streaming=True)
    print("Streaming slinusc/PubMedAbstractsSubset ...", flush=True)
    produced = 0
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_scan and produced >= target:
            break
        pmid = str(row.get("PMID") or row.get("pmid") or "").strip()
        if not pmid or pmid in excluded_pmids:
            continue
        title = str(row.get("title") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        body = f"{title}. {abstract}".strip() if title else abstract
        if len(body) < 120:
            continue
        if _leaks_benchmark(context_fps, body):
            continue
        prefix = f"pubmed_{pmid}"
        for ch in chunk_text_windows(
            body,
            source_prefix=prefix,
            title=title,
            section="abstract",
            pmid=pmid,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
        ):
            produced += 1
            yield ch
            if produced >= target:
                return
        if scanned % 50_000 == 0:
            print(f"  PubMed scan={scanned} chunks={produced}", flush=True)


def build_corpus(
    out_path: str,
    *,
    target_chunks: int = 25_000,
    chunk_words: int = 300,
    overlap_words: int = 54,
    min_words: int = 40,
    legacy_chunks: str = "",
    medquad_only: bool = False,
    pubmed_cap: int = 0,
) -> int:
    excluded = load_pubmedqa_excluded_pubids()
    if legacy_chunks:
        excluded |= load_legacy_pubmedqa_pmids(legacy_chunks)
    context_fps = load_context_fingerprints()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    written = 0
    stats: Dict[str, int] = {"medquad": 0, "pubmed": 0}

    with open(out_path, "w", encoding="utf-8") as fout:
        print("=== Source 1: MedQuad (NIH consumer health) ===", flush=True)
        for ch in iter_medquad_chunks(
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            min_words=min_words,
            context_fps=context_fps,
        ):
            fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
            written += 1
            stats["medquad"] += 1
            if written >= target_chunks:
                break
        print(f"  MedQuad chunks written: {stats['medquad']}", flush=True)

        if not medquad_only and written < target_chunks:
            need = target_chunks - written
            cap = pubmed_cap if pubmed_cap > 0 else need
            print(f"=== Source 2: PubMed abstracts (need ~{need}, cap {cap}) ===", flush=True)
            for ch in iter_pubmed_abstract_chunks(
                excluded,
                target=cap,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                min_words=min_words,
                context_fps=context_fps,
            ):
                fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                written += 1
                stats["pubmed"] += 1
                if written >= target_chunks:
                    break
            print(f"  PubMed abstract chunks written: {stats['pubmed']}", flush=True)

    print(
        f"\nWrote {written} chunks -> {out_path}\n"
        f"  medquad={stats['medquad']} pubmed={stats['pubmed']} excluded_pmids={len(excluded)}",
        flush=True,
    )
    if written < 15_000:
        print(
            "WARNING: < 15,000 chunks — increase --target_chunks or run without --medquad_only.",
            flush=True,
        )
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Build non-leaking external biomedical RAG corpus")
    p.add_argument(
        "--out",
        default=os.path.join("kaggle_working", "rag_index", "external_chunks.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument("--target_chunks", type=int, default=25_000, help="Target chunk count (15k-50k)")
    p.add_argument("--chunk_words", type=int, default=300, help="~256-384 tokens per chunk")
    p.add_argument("--overlap_words", type=int, default=54, help="~18%% overlap")
    p.add_argument("--min_words", type=int, default=40)
    p.add_argument(
        "--legacy_chunks",
        default=os.path.join("kaggle_working", "rag_index", "chunks.jsonl"),
        help="Exclude PMIDs from legacy pubmedqa_* sources in this file",
    )
    p.add_argument(
        "--medquad_only",
        action="store_true",
        help="Only MedQuad (~16k Q&A; may yield ~16-25k chunks with splitting)",
    )
    p.add_argument(
        "--pubmed_cap",
        type=int,
        default=0,
        help="Max chunks from PubMed stream (0 = fill to target)",
    )
    args = p.parse_args()

    if args.target_chunks < 1000:
        print("--target_chunks too small", file=sys.stderr)
        sys.exit(2)

    n = build_corpus(
        os.path.abspath(args.out),
        target_chunks=args.target_chunks,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words=args.min_words,
        legacy_chunks=args.legacy_chunks if args.legacy_chunks else "",
        medquad_only=args.medquad_only,
        pubmed_cap=args.pubmed_cap,
    )
    if n < 1:
        sys.exit(2)


if __name__ == "__main__":
    main()
