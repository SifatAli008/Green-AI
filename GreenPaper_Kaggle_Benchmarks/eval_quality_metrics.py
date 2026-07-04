"""
Quality metrics for clinical QA evaluation.
- F1 over token-level overlap (as in paper methodology).
- Optional: ROUGE/BLEU if packages available.
- Reference answers loaded from references.json; same tokenization for pred and ref.
"""
import json
import re
import os
from typing import AbstractSet, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Prefix length for matching predictions to references when keys are long clinical queries.
QUERY_TRUNC_LEN = 500

# Default tokenization: whitespace + punctuation split (reproducible without transformers)
def tokenize_for_f1(text: str) -> List[str]:
    """Tokenize for F1: lowercase, split on non-alphanumeric, keep non-empty."""
    if not text or not isinstance(text, str):
        return []
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens

def token_level_precision_recall_f1(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float]:
    """
    Precision = |pred ∩ ref| / |pred|
    Recall    = |pred ∩ ref| / |ref|
    F1        = 2 * P * R / (P + R)
    Count matches by aligning ref tokens; each ref token can match at most one pred token.
    """
    if not ref_tokens:
        return (0.0, 0.0, 0.0) if pred_tokens else (1.0, 1.0, 1.0)
    if not pred_tokens:
        return (1.0, 0.0, 0.0)
    ref_set = list(ref_tokens)  # order for "first match"
    pred_list = list(pred_tokens)
    matches = 0
    used = set()
    for r in ref_set:
        for i, p in enumerate(pred_list):
            if i not in used and p == r:
                used.add(i)
                matches += 1
                break
    prec = matches / len(pred_list) if pred_list else 0.0
    rec = matches / len(ref_set) if ref_set else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return (prec, rec, f1)

def compute_f1(prediction: str, reference: str, tokenizer=None) -> Dict[str, float]:
    """
    Compute token-level P, R, F1 between prediction and reference.
    tokenizer: optional callable (str -> list of str). Default: tokenize_for_f1.
    """
    tok = tokenizer or tokenize_for_f1
    p_tok = tok(prediction)
    r_tok = tok(reference)
    prec, rec, f1 = token_level_precision_recall_f1(p_tok, r_tok)
    return {"precision": prec, "recall": rec, "f1": f1, "pred_len": len(p_tok), "ref_len": len(r_tok)}

def doc_matches_relevant(doc_id: str, relevant_id: str) -> bool:
    """
    Match exact source or chunked suffix (``pubmedqa_123`` vs ``pubmedqa_123_0``).
    """
    doc_id = str(doc_id or "")
    relevant_id = str(relevant_id or "")
    if not doc_id or not relevant_id:
        return False
    if doc_id == relevant_id:
        return True
    return doc_id.startswith(relevant_id + "_")


def _relevant_hit_in_list(ranked: Sequence[str], relevant_ids: AbstractSet[str]) -> bool:
    rel = {str(x) for x in relevant_ids if str(x)}
    for doc in ranked:
        for r in rel:
            if doc_matches_relevant(str(doc), r):
                return True
    return False


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_ids: AbstractSet[str],
    k: int,
) -> float:
    """
    Fraction of relevant documents found in the top-``k`` ranked results.

    Supports prefix match: gold ``pubmedqa_<pubid>`` matches ``pubmedqa_<pubid>_0``, etc.
  """
    rel = {str(x) for x in relevant_ids if str(x)}
    if not rel:
        return float("nan")
    k = max(1, int(k))
    top = [str(x) for x in ranked_doc_ids[:k] if str(x)]
    found = sum(1 for gold in rel if _relevant_hit_in_list(top, {gold}))
    return float(found) / float(len(rel))


def mrr(
    ranked_doc_ids: Sequence[str],
    relevant_ids: AbstractSet[str],
) -> float:
    """Mean reciprocal rank: 1/rank of the first relevant doc, or 0 if none appear."""
    rel = {str(x) for x in relevant_ids if str(x)}
    if not rel:
        return float("nan")
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        for gold in rel:
            if doc_matches_relevant(str(doc_id), gold):
                return 1.0 / float(rank)
    return 0.0


def compute_retrieval_metrics(
    ranked_doc_ids: Sequence[str],
    relevant_ids: Union[AbstractSet[str], Iterable[str]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Per-query Recall@K and MRR for one ranked retrieval list."""
    rel = set(relevant_ids) if not isinstance(relevant_ids, set) else relevant_ids
    ranked = [str(x) for x in ranked_doc_ids if str(x)]
    out: Dict[str, float] = {"mrr": mrr(ranked, rel)}
    for k in ks:
        kk = max(1, int(k))
        out[f"recall_at_{kk}"] = recall_at_k(ranked, rel, kk)
    return out


def mean_retrieval_metrics(
    per_query: Sequence[Dict[str, float]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Optional[float]]:
    """Macro-average of per-query retrieval metrics (skips NaN)."""
    if not per_query:
        return {}
    out: Dict[str, Optional[float]] = {}

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in per_query if key in r and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else None

    out["mrr_mean"] = _mean("mrr")
    for k in ks:
        out[f"recall_at_{max(1, int(k))}_mean"] = _mean(f"recall_at_{max(1, int(k))}")
    out["n_queries"] = len(per_query)
    return out


def load_references(path: str = "references.json") -> Dict[str, str]:
    """Load {query_id: reference_answer} from JSON. query_id can be question text or id."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_with_references(
    predictions: List[Dict],
    references: Dict[str, str],
    query_key: str = "query",
    pred_key: str = "response"
) -> Dict:
    """
    predictions: list of {"query": "...", "response": "...", ...}
    references: {query or query_id: reference_text}
    Returns: per-sample F1 list, mean F1, and details.
    """
    results = []
    for item in predictions:
        q = item.get(query_key, "")
        pred = item.get(pred_key, "")
        ref = references.get(q, references.get(q[:QUERY_TRUNC_LEN], ""))  # allow truncation match
        if not ref:
            results.append({"query": q[:QUERY_TRUNC_LEN], "f1": None, "reason": "no_reference"})
            continue
        m = compute_f1(pred, ref)
        m["query"] = q[:QUERY_TRUNC_LEN]
        results.append(m)
    f1s = [r["f1"] for r in results if r.get("f1") is not None]
    return {
        "mean_f1": sum(f1s) / len(f1s) if f1s else None,
        "std_f1": (sum((x - sum(f1s)/len(f1s))**2 for x in f1s)**0.5 / len(f1s)) if len(f1s) > 1 else 0.0,
        "n_with_ref": len(f1s),
        "n_total": len(predictions),
        "per_sample": results,
    }

if __name__ == "__main__":
    # Quick test
    ranked = ["pubmedqa_b", "pubmedqa_c", "pubmedqa_a"]
    rel = {"pubmedqa_a"}
    print("Recall@3:", recall_at_k(ranked, rel, 3))
    print("MRR:", mrr(ranked, rel))
    ref = "Metformin is first line therapy for type 2 diabetes. Lifestyle modification is recommended."
    pred = "Metformin is the first line treatment for diabetes. Lifestyle changes are also recommended."
    out = compute_f1(pred, ref)
    print("Token-level F1 test:", out)
    # Test with references file if present
    refs = load_references()
    print("Loaded references:", len(refs), "entries")
