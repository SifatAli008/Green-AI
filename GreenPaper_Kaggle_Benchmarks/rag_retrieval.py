"""
RAG retrieval repair pipeline (BIOMEDICAL RAG PIPELINE — Retrieval Repair Plan).

Dense (FAISS IndexFlatIP + L2-normalized vectors) + BM25 + RRF + optional cross-encoder rerank,
context cleaning, and per-query diagnostics for eval_benchmarks.
"""
from __future__ import annotations

import heapq
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

LAST_RETRIEVAL_DIAGNOSTIC: Dict[str, Any] = {}
_CROSS_ENCODER_CACHE: Dict[str, Any] = {}
_BM25_CACHE: Dict[int, Any] = {}
_HYBRID_TIMING_LOGGED: bool = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def reciprocal_rank_fusion(
    dense_ids: Sequence[int],
    bm25_ids: Sequence[int],
    *,
    k: int = 60,
) -> List[int]:
    """RRF merge of two ranked doc-id lists (Step 04)."""
    scores: Dict[int, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[int(doc_id)] = scores.get(int(doc_id), 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


class SimpleBM25:
    """Lightweight Okapi BM25 over in-memory corpus (no rank_bm25 dependency)."""

    def __init__(self, texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.doc_tfs: List[Counter[str]] = []
        df: Counter[str] = Counter()
        for raw in texts:
            toks = tokenize(raw)
            self.doc_lens.append(len(toks))
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            for t in tf:
                df[t] += 1
        self.n_docs = len(texts)
        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0
        self.df = df

    def score(self, query: str, doc_idx: int) -> float:
        if doc_idx < 0 or doc_idx >= self.n_docs:
            return 0.0
        q_toks = tokenize(query)
        if not q_toks:
            return 0.0
        tf = self.doc_tfs[doc_idx]
        dl = self.doc_lens[doc_idx]
        if dl == 0:
            return 0.0
        s = 0.0
        for t in q_toks:
            if t not in self.df:
                continue
            idf = math.log(1.0 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            freq = tf.get(t, 0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s += idf * (freq * (self.k1 + 1.0)) / max(denom, 1e-9)
        return s

    def top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        n = max(1, min(n, self.n_docs))
        heap: List[Tuple[float, int]] = []
        for i in range(self.n_docs):
            sc = self.score(query, i)
            if sc <= 0:
                continue
            if len(heap) < n:
                heapq.heappush(heap, (sc, i))
            elif sc > heap[0][0]:
                heapq.heapreplace(heap, (sc, i))
        return [(i, sc) for sc, i in sorted(heap, reverse=True)]


def validate_embedding_matrix(mat: Any) -> None:
    """Step 02 validation: shape, NaN, zero rows, L2 norms ≈ 1."""
    import numpy as np

    if mat.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {mat.shape}")
    if int(np.isnan(mat).sum()) > 0:
        raise ValueError("embeddings contain NaN")
    norms = np.linalg.norm(mat, axis=1)
    if not (norms > 0).all():
        raise ValueError("embeddings contain zero vectors")
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise ValueError(
            f"L2 norms not ≈1 after normalize (min={float(norms.min()):.4f}, max={float(norms.max()):.4f})"
        )


def clean_passage_text(text: str) -> str:
    """Strip internal tags; keep biomedical prose (Step 06)."""
    t = (text or "").strip()
    t = re.sub(r"\b(idx|source|dense|lexical|combined)=[^\s,]+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def dedupe_sentences(passages: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for p in passages:
        parts = _SENT_SPLIT.split(clean_passage_text(p))
        kept: List[str] = []
        for sent in parts:
            key = sent.strip().lower()
            if len(key) < 12 or key in seen:
                continue
            seen.add(key)
            kept.append(sent.strip())
        if kept:
            out.append(" ".join(kept))
    return out


def format_biomedical_evidence(
    passages: Sequence[str],
    query: str,
    *,
    max_chars: int,
) -> str:
    """Step 06 prompt block (evidence only; model wrapper adds instruction)."""
    cleaned = dedupe_sentences(passages)
    lines: List[str] = []
    used = 0
    for i, p in enumerate(cleaned, start=1):
        block = f"[{i}] {p}"
        if used + len(block) + 2 > max_chars:
            remain = max_chars - used - 20
            if remain > 80:
                block = f"[{i}] {p[:remain]}..."
            else:
                break
        lines.append(block)
        used += len(block) + 2
    if not lines:
        return ""
    header = (
        "You are a biomedical expert. Answer only from the evidence below. "
        "If evidence is insufficient, say so explicitly.\n\n"
        "Retrieved Medical Evidence:\n"
    )
    body = "\n".join(lines)
    footer = f"\n\nQuestion: {query.strip()}\n\nAnswer using evidence references [1][2][3] where applicable."
    return header + body + footer


def clear_retrieval_caches() -> None:
    """Drop BM25 cache when the chunk corpus object changes (keep cross-encoder)."""
    _BM25_CACHE.clear()


def clear_all_retrieval_caches() -> None:
    """Drop BM25 + cross-encoder caches (rare; corpus or reranker model change)."""
    _CROSS_ENCODER_CACHE.clear()
    _BM25_CACHE.clear()


def clear_cross_encoder_cache() -> None:
    """Backward-compatible alias."""
    clear_all_retrieval_caches()


def _bm25_for_corpus(texts: Sequence[str]) -> SimpleBM25:
    """Build BM25 once per in-memory corpus (25k-chunk index: major speedup)."""
    key = id(texts)
    bm = _BM25_CACHE.get(key)
    if bm is None:
        bm = SimpleBM25(texts)
        _BM25_CACHE[key] = bm
        if not _env_truthy("RAG_QUIET"):
            print(f"RAG: BM25 index built for {len(texts)} chunks (cached).", flush=True)
    return bm


def _load_cross_encoder(model_name: str) -> Any:
    """Load cross-encoder once per process (prefer real_model_runner global cache)."""
    name = (model_name or "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[name]
    try:
        import real_model_runner as rmr

        ce = rmr.get_cached_cross_encoder(name)
        _CROSS_ENCODER_CACHE[name] = ce
        return ce
    except Exception:
        pass
    from sentence_transformers import CrossEncoder

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    ce = CrossEncoder(name, max_length=512, device="cpu")
    _CROSS_ENCODER_CACHE[name] = ce
    return ce


def cross_encoder_rerank(
    query: str,
    candidate_ids: Sequence[int],
    texts: Sequence[str],
    *,
    model_name: str,
    top_k: int,
) -> List[Tuple[int, float]]:
    if not candidate_ids:
        return []
    pairs = [(query, texts[int(i)]) for i in candidate_ids if 0 <= int(i) < len(texts)]
    if not pairs:
        return []
    try:
        ce = _load_cross_encoder(model_name)
        scores = ce.predict(
            pairs,
            batch_size=min(32, max(1, len(pairs))),
            show_progress_bar=False,
        )
    except Exception:
        return [(int(i), 0.0) for i in candidate_ids[:top_k]]
    ranked = sorted(
        zip(candidate_ids, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(int(i), float(sc)) for i, sc in ranked[:top_k]]


def hybrid_retrieve(
    query: str,
    *,
    index: Any,
    texts: Sequence[str],
    sources: Sequence[str],
    embedder: Any,
    top_k: int = 3,
    max_context_chars: int = 2000,
    metrics_max_k: int = 50,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Full repair pipeline: dense@K + BM25@K → RRF → rerank → clean → format.

    Returns (evidence_block, final_hits, ranked_metrics_hits, method, diagnostic).
    """
    global LAST_RETRIEVAL_DIAGNOSTIC
    import faiss  # type: ignore
    import numpy as np

    t0 = time.perf_counter()
    ntotal = int(index.ntotal)
    rrf_k = _env_int("RAG_RRF_K", 60)
    fetch_n = _env_int("RAG_RERANK_CANDIDATES", 20)
    fetch_n = max(fetch_n, top_k)
    min_dense = _env_float("RAG_MIN_DENSE_SCORE", 0.35)
    use_rrf = not _env_truthy("RAG_DISABLE_RRF")
    use_rerank = not _env_truthy("RAG_DISABLE_RERANK")
    reranker_model = os.environ.get(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ).strip()

    try:
        try:
            qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            qv = embedder.encode([query], convert_to_numpy=True)
        if qv.dtype != np.float32:
            qv = qv.astype(np.float32)
        faiss.normalize_L2(qv)
    except Exception as ex:
        diag = {"error": str(ex), "retrieval_method": "dense_failed"}
        LAST_RETRIEVAL_DIAGNOSTIC = diag
        return None, [], [], "dense_failed", diag

    dense_k = min(fetch_n, ntotal, max(1, ntotal))
    sims, ids = index.search(qv, dense_k)
    dense_ids: List[int] = []
    dense_scores: Dict[int, float] = {}
    for rank, j in enumerate(ids[0]):
        if j < 0 or j >= len(texts):
            continue
        jj = int(j)
        sc = float(sims[0][rank])
        if sc != sc:
            sc = 0.0
        dense_ids.append(jj)
        dense_scores[jj] = sc

    bm25 = _bm25_for_corpus(texts)
    bm25_ranked = bm25.top_n(query, fetch_n)
    bm25_ids = [i for i, _ in bm25_ranked]
    bm25_scores = {i: sc for i, sc in bm25_ranked}

    if use_rrf and bm25_ids:
        fused_ids = reciprocal_rank_fusion(dense_ids, bm25_ids, k=rrf_k)
        method = "hybrid"
    else:
        fused_ids = dense_ids or bm25_ids
        method = "dense" if dense_ids else "bm25"

    # Filter by dense similarity before rerank
    filtered = [i for i in fused_ids if dense_scores.get(i, 0.0) >= min_dense or i in bm25_scores]
    if not filtered:
        filtered = fused_ids[:fetch_n]

    candidate_ids = filtered[:fetch_n]
    reranker_scores: Dict[int, float] = {}
    if use_rerank and len(candidate_ids) > top_k:
        reranked = cross_encoder_rerank(
            query, candidate_ids, texts, model_name=reranker_model, top_k=top_k
        )
        method = "hybrid+rerank" if use_rrf else "dense+rerank"
        final_ids = [i for i, _ in reranked]
        reranker_scores = {i: sc for i, sc in reranked}
    else:
        final_ids = candidate_ids[:top_k]

    ranked_metrics: List[Dict[str, Any]] = []
    for i in fused_ids[:metrics_max_k]:
        ranked_metrics.append(
            {
                "idx": i,
                "source": str(sources[i]) if i < len(sources) else "",
                "dense_score": dense_scores.get(i),
                "bm25_score": bm25_scores.get(i),
            }
        )

    final_hits: List[Dict[str, Any]] = []
    passages: List[str] = []
    for rank, i in enumerate(final_ids, start=1):
        body = clean_passage_text(str(texts[i]))
        if not body:
            continue
        passages.append(body)
        final_hits.append(
            {
                "rank": rank,
                "idx": i,
                "source": str(sources[i]) if i < len(sources) else "",
                "body": body,
                "dense_score": dense_scores.get(i),
                "bm25_score": bm25_scores.get(i),
                "reranker_score": reranker_scores.get(i),
            }
        )

    block = format_biomedical_evidence(passages, query, max_chars=max_context_chars)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    ranked_source_ids = [
        str(h.get("source") or f"idx_{h.get('idx')}")
        for h in ranked_metrics
        if h.get("source") or h.get("idx") is not None
    ]
    diag: Dict[str, Any] = {
        "query": query[:500],
        "retrieved_chunk_ids": ranked_source_ids
        or [str(sources[i]) if i < len(sources) else f"idx_{i}" for i in candidate_ids],
        "similarity_scores": [dense_scores.get(i) for i in candidate_ids],
        "bm25_scores": [bm25_scores.get(i) for i in candidate_ids],
        "reranker_scores": [reranker_scores.get(i) for i in final_ids],
        "final_chunk_ids": [h.get("source") or f"idx_{h['idx']}" for h in final_hits],
        "retrieval_method": method,
        "context_char_length": len(block),
        "retrieval_latency_ms": round(latency_ms, 2),
    }
    LAST_RETRIEVAL_DIAGNOSTIC = diag
    if not block:
        return None, [], ranked_metrics, method, diag
    return block, final_hits, ranked_metrics, method, diag


def retrieval_health_summary(
    per_query_metrics: Sequence[Dict[str, float]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """Step 08: detect binary collapse (recall@1 == recall@3 == ...)."""
    from eval_quality_metrics import mean_retrieval_metrics

    means = mean_retrieval_metrics(list(per_query_metrics), ks=ks)
    collapse = True
    r1 = means.get("recall_at_1_mean")
    for k in ks[1:]:
        rk = means.get(f"recall_at_{k}_mean")
        if r1 is None or rk is None:
            collapse = False
            break
        if abs(float(r1) - float(rk)) > 1e-6:
            collapse = False
            break
    mrr_m = means.get("mrr_mean")
    mrr_lt_r1 = False
    if mrr_m is not None and r1 is not None and mrr_m == mrr_m and r1 == r1:
        mrr_lt_r1 = float(mrr_m) < float(r1) - 1e-6
    return {
        **means,
        "binary_collapse_detected": collapse,
        "mrr_below_recall_at_1": mrr_lt_r1,
        "healthy": (not collapse) and (mrr_lt_r1 or (r1 is not None and float(r1) < 0.99)),
    }
