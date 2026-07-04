"""
Quality metrics for clinical QA evaluation.
- F1 over token-level overlap (as in paper methodology).
- Optional: ROUGE/BLEU if packages available.
- Reference answers loaded from references.json; same tokenization for pred and ref.
"""
import json
import re
import os
from typing import List, Dict, Tuple, Optional

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
        ref = references.get(q, references.get(q[:80], ""))  # allow truncation match
        if not ref:
            results.append({"query": q[:60], "f1": None, "reason": "no_reference"})
            continue
        m = compute_f1(pred, ref)
        m["query"] = q[:60]
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
    ref = "Metformin is first line therapy for type 2 diabetes. Lifestyle modification is recommended."
    pred = "Metformin is the first line treatment for diabetes. Lifestyle changes are also recommended."
    out = compute_f1(pred, ref)
    print("Token-level F1 test:", out)
    # Test with references file if present
    refs = load_references()
    print("Loaded references:", len(refs), "entries")
