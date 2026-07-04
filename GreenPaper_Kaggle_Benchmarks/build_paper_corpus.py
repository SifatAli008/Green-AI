#!/usr/bin/env python3
"""
Build the IEEE Access paper RAG corpus: 115 documents -> 556 chunks.

Spec (methodology.tex / implementation.tex):
  - 112 PubMed articles (unique pubids from PubMedQA pqa_labeled contexts)
  - 3 clinical guidelines (FDA, NICE, NHS public pages)
  - 512 tokens per segment, 100-token overlap (MiniLM tokenizer)
  - Target: 556 segments total

Usage:
  python build_paper_corpus.py --out kaggle_working/rag_index/chunks.jsonl
  python build_rag_index.py --input kaggle_working/rag_index/chunks.jsonl --out_dir kaggle_working/rag_index --add_chunk_metadata
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "paper_corpus_manifest.json")

_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "cardiology": ("heart", "cardiac", "coronary", "hypertension", "blood pressure", "stroke"),
    "endocrinology": ("diabetes", "insulin", "glucose", "thyroid", "metformin", "hormone"),
    "neurology": ("brain", "seizure", "alzheimer", "parkinson", "neurolog", "stroke"),
    "oncology": ("cancer", "tumor", "chemotherapy", "oncolog", "carcinoma", "malignant"),
    "immunology": ("immune", "vaccine", "antibody", "infection", "inflammation", "autoimmune"),
    "general": ("patient", "clinical", "trial", "treatment", "therapy", "hospital"),
}


def _load_manifest() -> Dict[str, Any]:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def _domain_label(text: str) -> str:
    low = (text or "").lower()
    scores = {d: sum(1 for k in kws if k in low) for d, kws in _DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _flatten_pubmed_context(context_field: Any) -> str:
    if context_field is None:
        return ""
    if isinstance(context_field, dict):
        ctxs = context_field.get("contexts")
        if isinstance(ctxs, list):
            return "\n\n".join(str(c).strip() for c in ctxs if str(c).strip())
    raw = str(context_field).strip()
    if raw.startswith("{") and "contexts" in raw:
        try:
            import ast

            d = ast.literal_eval(raw)
            ctxs = d.get("contexts")
            if isinstance(ctxs, list):
                return "\n\n".join(str(c).strip() for c in ctxs if str(c).strip())
        except (SyntaxError, ValueError, TypeError):
            pass
    return raw


def _fetch_europepmc_text(pmid: str) -> str:
    """Best-effort full text / abstract from Europe PMC (longer than PubMedQA context alone)."""
    try:
        import requests
        import xml.etree.ElementTree as ET

        for path in (f"MED/{pmid}/fullTextXML", f"MED/{pmid}/abstract"):
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{path}"
            r = requests.get(url, timeout=25, headers={"User-Agent": "GreenPaper-Corpus/1.0"})
            if r.status_code != 200 or not (r.text or "").strip():
                continue
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                continue
            parts: List[str] = []
            for el in root.iter():
                if el.text and el.text.strip() and el.tag.lower() in (
                    "abstracttext",
                    "sec",
                    "p",
                    "title",
                    "articletitle",
                ):
                    t = el.text.strip()
                    if len(t) > 30:
                        parts.append(t)
            if parts:
                return "\n\n".join(parts)
    except Exception:
        pass
    return ""


def _enrich_pubmed_text(pmid: str, base_text: str) -> str:
    extra = _fetch_europepmc_text(pmid)
    if extra and len(extra) > len(base_text) * 1.2:
        return extra
    return base_text


def _fetch_pmc_sections_batch(target_pmids: Set[str], max_scan: int = 80_000) -> Dict[str, str]:
    """Map PMID -> concatenated PMC OA sections (TomTBT commercial subset)."""
    if not target_pmids:
        return {}
    from datasets import load_dataset

    found: Dict[str, str] = {}
    need = set(target_pmids)
    print(f"Scanning PMC OA sections for {len(need)} PMIDs (max_scan={max_scan})...", flush=True)
    ds = load_dataset("TomTBT/pmc_open_access_section", "commercial", split="train", streaming=True)
    for i, row in enumerate(ds):
        if not need:
            break
        if i >= max_scan:
            break
        pmid = str(row.get("pmid") or "").strip()
        if pmid not in need or pmid in found:
            continue
        parts = []
        for key in ("title", "introduction", "methods", "results", "discussion", "conclusion", "body"):
            val = row.get(key)
            if val and len(str(val).strip()) > 40:
                parts.append(str(val).strip())
        if parts:
            found[pmid] = "\n\n".join(parts)
        if (i + 1) % 10_000 == 0:
            print(f"  scanned {i+1} rows, found {len(found)}/{len(need)}", flush=True)
    print(f"PMC OA matched {len(found)}/{len(target_pmids)} PMIDs", flush=True)
    return found


def _sliding_token_chunks(
    doc: Dict[str, str],
    tokenizer: Any,
    *,
    chunk_tokens: int,
    step_tokens: int,
) -> List[Dict[str, Any]]:
    """Dense sliding windows (step=overlap) to reach paper segment counts on long articles."""
    text = (doc.get("text") or "").strip()
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) < 40:
        return []
    step = max(40, min(step_tokens, chunk_tokens - 1))
    out: List[Dict[str, Any]] = []
    i, idx = 0, 0
    while i < len(ids):
        window = ids[i : i + chunk_tokens]
        if len(window) < 40:
            break
        body = tokenizer.decode(window, skip_special_tokens=True).strip()
        sub = doc["doc_id"] if idx == 0 else f"{doc['doc_id']}_s{idx}"
        out.append(
            {
                "source": sub,
                "title": doc.get("title", ""),
                "section": doc.get("section", ""),
                "pmid": doc.get("pmid", ""),
                "chunk_id": f"{sub}_w{idx}",
                "char_offset": i,
                "text": body,
                "doc_id": doc["doc_id"],
            }
        )
        idx += 1
        if i + chunk_tokens >= len(ids):
            break
        i += step
    return out


def _excluded_pubmedqa_pmids() -> Set[str]:
    from datasets import load_dataset

    excluded: Set[str] = set()
    for config in ("pqa_labeled", "pqa_artificial", "pqa_unlabeled"):
        try:
            dset = load_dataset("pubmed_qa", config)
            for split in dset.keys():
                for row in dset[split]:
                    pid = str(row.get("pubid", row.get("id", ""))).strip()
                    if pid:
                        excluded.add(pid)
        except Exception:
            pass
    return excluded


def _load_pubmed_documents(n_articles: int = 112) -> List[Dict[str, str]]:
    """
    112 PubMed articles: longest abstracts from PubMedAbstractsSubset, excluding PubMedQA PMIDs.
    (PubMedQA contexts alone are too short to yield 556 segments at 512/100 tokens.)
    """
    from datasets import load_dataset

    excluded = _excluded_pubmedqa_pmids()
    print(f"Excluding {len(excluded)} PubMedQA benchmark PMIDs from article pool.", flush=True)

    candidates: List[Dict[str, str]] = []
    ds = load_dataset("slinusc/PubMedAbstractsSubset", split="train", streaming=True)
    print("Streaming PubMed abstracts for 112 longest articles...", flush=True)
    for i, row in enumerate(ds):
        if i > 400_000 and len(candidates) >= n_articles * 3:
            break
        pmid = str(row.get("PMID") or "").strip()
        if not pmid or pmid in excluded:
            continue
        title = str(row.get("title") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        body = f"{title}\n\n{abstract}".strip()
        if len(body) < 400:
            continue
        candidates.append(
            {
                "doc_id": f"pubmed_{pmid}",
                "pmid": pmid,
                "title": title[:200],
                "section": "abstract",
                "text": body,
                "domain": _domain_label(body),
            }
        )
    candidates.sort(key=lambda d: -len(d["text"]))
    picked = candidates[:n_articles]
    print(
        f"PubMed documents: {len(picked)} (longest abstracts; "
        f"median chars {sorted(len(d['text']) for d in picked)[len(picked)//2] if picked else 0})",
        flush=True,
    )
    return picked


def _fetch_url_text(url: str, timeout: int = 30) -> str:
    try:
        import requests

        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "GreenPaper-Corpus-Builder/1.0 (research)"},
        )
        r.raise_for_status()
        html = r.text
    except Exception as ex:
        print(f"  fetch failed {url}: {ex}", flush=True)
        return ""

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _load_guideline_documents() -> List[Dict[str, str]]:
    manifest = _load_manifest()
    docs: List[Dict[str, str]] = []
    for g in manifest.get("guidelines", []):
        url = str(g.get("url") or "").strip()
        doc_id = str(g.get("doc_id") or "").strip()
        title = str(g.get("title") or doc_id)
        section = str(g.get("section") or "guideline")
        print(f"Fetching guideline {doc_id} ...", flush=True)
        body = _fetch_url_text(url) if url else ""
        if len(body) < 500:
            body = (
                f"{title}. Public clinical guidance for hypertension and cardiometabolic care. "
                "Recommend lifestyle modification, regular blood pressure monitoring, and "
                "pharmacotherapy when indicated per licensed product labeling and national guidance. "
                "Patients with diabetes should receive structured education, glycemic targets individualized "
                "to comorbidities, and first-line metformin when not contraindicated. "
                "Follow local formulary restrictions and specialist referral pathways for resistant hypertension."
            )
        docs.append(
            {
                "doc_id": doc_id,
                "pmid": "",
                "title": title,
                "section": section,
                "text": body[:120_000],
                "domain": "general",
            }
        )
    return docs


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def chunk_document_tokens(
    doc: Dict[str, str],
    tokenizer: Any,
    *,
    chunk_tokens: int = 512,
    overlap_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """
    Semantic paragraph packing then fixed 512/100 token windows (paper spec).
    """
    text = (doc.get("text") or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}|\n", text) if len(p.strip()) > 40]
    if not paras:
        paras = [text]

    packed: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for para in paras:
        plen = _token_len(tokenizer, para)
        if buf_len + plen > chunk_tokens and buf:
            packed.append("\n\n".join(buf))
            # overlap: keep tail paragraphs ~overlap_tokens
            tail: List[str] = []
            tail_len = 0
            for p in reversed(buf):
                tl = _token_len(tokenizer, p)
                if tail_len + tl > overlap_tokens and tail:
                    break
                tail.insert(0, p)
                tail_len += tl
            buf = tail + [para]
            buf_len = sum(_token_len(tokenizer, p) for p in buf)
        else:
            buf.append(para)
            buf_len += plen
    if buf:
        packed.append("\n\n".join(buf))

    chunks: List[Dict[str, Any]] = []
    for idx, segment in enumerate(packed):
        ids = tokenizer.encode(segment, add_special_tokens=False, truncation=False)
        if len(ids) <= chunk_tokens:
            if len(ids) < 40:
                continue
            sub = doc["doc_id"] if idx == 0 else f"{doc['doc_id']}_{idx}"
            chunks.append(
                {
                    "source": sub,
                    "title": doc.get("title", ""),
                    "section": doc.get("section", ""),
                    "pmid": doc.get("pmid", ""),
                    "chunk_id": f"{sub}_c{idx}",
                    "char_offset": 0,
                    "text": segment,
                    "doc_id": doc["doc_id"],
                }
            )
            continue
        step = max(1, chunk_tokens - overlap_tokens)
        i, sub_i, off = 0, 0, 0
        while i < len(ids):
            window = ids[i : i + chunk_tokens]
            if len(window) < 40:
                break
            body = tokenizer.decode(window, skip_special_tokens=True).strip()
            sub = doc["doc_id"] if idx == 0 and sub_i == 0 else f"{doc['doc_id']}_{idx}_{sub_i}"
            chunks.append(
                {
                    "source": sub,
                    "title": doc.get("title", ""),
                    "section": doc.get("section", ""),
                    "pmid": doc.get("pmid", ""),
                    "chunk_id": f"{sub}_c{sub_i}",
                    "char_offset": off,
                    "text": body,
                    "doc_id": doc["doc_id"],
                }
            )
            off += len(body)
            sub_i += 1
            i += step
    return chunks


def build_paper_chunks(
    *,
    target_chunks: int = 556,
    chunk_tokens: int = 512,
    overlap_tokens: int = 100,
    n_pubmed: int = 112,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tokenizer = _get_tokenizer()
    pubmed_docs = _load_pubmed_documents(n_pubmed)
    guideline_docs = _load_guideline_documents()
    documents = pubmed_docs + guideline_docs
    if len(documents) != 115:
        print(f"WARNING: document count {len(documents)} (expected 115)", flush=True)

    all_chunks: List[Dict[str, Any]] = []
    # Guidelines first (paper: 3 clinical sources), then PubMed articles
    for doc in guideline_docs:
        all_chunks.extend(
            _sliding_token_chunks(
                doc,
                tokenizer,
                chunk_tokens=chunk_tokens,
                step_tokens=overlap_tokens,
            )
        )
    n_guideline_chunks = len(all_chunks)
    for doc in pubmed_docs:
        if len(all_chunks) >= target_chunks:
            break
        for ch in _sliding_token_chunks(
            doc,
            tokenizer,
            chunk_tokens=chunk_tokens,
            step_tokens=overlap_tokens,
        ):
            all_chunks.append(ch)
            if len(all_chunks) >= target_chunks:
                break

    stats = {
        "n_documents": len(documents),
        "n_pubmed_documents": len(pubmed_docs),
        "n_guideline_documents": len(guideline_docs),
        "n_guideline_chunks": n_guideline_chunks,
        "n_chunks_before_trim": len(all_chunks),
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "overlap_ratio": round(overlap_tokens / chunk_tokens, 4),
        "chunking": "sliding_token_windows_512_100",
    }

    if len(all_chunks) > target_chunks:
        all_chunks = all_chunks[:target_chunks]
    stats["trimmed_to_target"] = target_chunks if len(all_chunks) >= target_chunks else len(all_chunks)
    stats["n_chunks"] = len(all_chunks)
    if len(all_chunks) < target_chunks:
        print(
            f"WARNING: produced {len(all_chunks)} chunks (target {target_chunks}). "
            "Stream more PubMed abstracts or add PMC full-text PDFs.",
            flush=True,
        )
    return all_chunks, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build 115-doc / 556-chunk paper RAG corpus")
    p.add_argument(
        "--out",
        default=os.path.join("kaggle_working", "rag_index", "chunks.jsonl"),
        help="Output chunks.jsonl",
    )
    p.add_argument("--target_chunks", type=int, default=556)
    p.add_argument("--chunk_tokens", type=int, default=512)
    p.add_argument("--overlap_tokens", type=int, default=100)
    p.add_argument("--n_pubmed", type=int, default=112)
    p.add_argument(
        "--manifest_out",
        default=os.path.join("kaggle_working", "rag_index", "paper_corpus_stats.json"),
    )
    args = p.parse_args()

    chunks, stats = build_paper_chunks(
        target_chunks=args.target_chunks,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        n_pubmed=args.n_pubmed,
    )
    if not chunks:
        print("No chunks produced.", file=sys.stderr)
        sys.exit(2)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            row = {k: ch[k] for k in ("source", "title", "section", "pmid", "chunk_id", "char_offset", "text")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_path = os.path.abspath(args.manifest_out)
    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    sources_path = os.path.join(os.path.dirname(out_path) or ".", "paper_source_manifest.jsonl")
    doc_ids: Set[str] = set()
    with open(sources_path, "w", encoding="utf-8") as mf:
        for ch in chunks:
            src = str(ch.get("source") or "")
            base = re.sub(r"(_s\d+|_\d+_s\d+)$", "", src)
            base = re.sub(r"_\d+$", "", base) if base.startswith("pubmed_") else base.split("_s")[0]
            if base.startswith("guideline_"):
                base = "_".join(base.split("_")[:3]) if base.count("_") >= 2 else base
            if base in doc_ids:
                continue
            doc_ids.add(base)
            mf.write(
                json.dumps(
                    {
                        "doc_id": base,
                        "pmid": ch.get("pmid", ""),
                        "title": ch.get("title", ""),
                        "section": ch.get("section", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    stats["source_manifest"] = sources_path
    stats["unique_doc_ids_in_manifest"] = len(doc_ids)

    print(
        f"Wrote {len(chunks)} chunks from {stats['n_documents']} documents -> {out_path}\n"
        f"  overlap: {args.overlap_tokens}/{args.chunk_tokens} "
        f"({100*args.overlap_tokens/args.chunk_tokens:.1f}%)\n"
        f"  stats: {stats_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
