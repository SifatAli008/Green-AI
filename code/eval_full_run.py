"""
Full evaluation run: expanded query set, all metrics (F1, safety, energy, routing).
- Loads queries from queries_expanded.json (or fallback 12).
- Runs 2x2 (SLM/LLM x RAG/NoRAG) on a subset or full set; collects responses.
- Computes: F1 vs references, safety claim counts, energy/carbon, latency.
- Saves results to eval_results.json and prints summary.
Run after: pip install torch transformers (and optionally sentence-transformers, faiss-cpu).
For GPU runs, use real_model_runner via EVAL_USE_REAL_MODEL; default is lightweight mock inference.
"""
import json
import time
import os
from typing import List, Dict, Any
from collections import defaultdict

# Optional: real inference (requires torch + transformers; HF token in env)
def _mock_inference(query: str, model_name: str, use_rag: bool) -> Dict[str, Any]:
    """Mock response when no GPU or skip heavy load."""
    evidence = "Metformin is first-line. Lifestyle modification recommended. " * 2 if use_rag else ""
    n_tok = 60 + (40 if use_rag else 0)
    return {
        "response": evidence + f" Summary for: {query[:50]}...",
        "response_tokens": n_tok,
        "latency_seconds": 0.5,
        "model": model_name,
        "rag": use_rag,
        "evidence": evidence[:200],
    }


def _real_inference(query: str, model_name: str, use_rag: bool, models_dict: dict) -> Dict[str, Any]:
    """Call real_model_runner for one query/model/rag. models_dict from real_model_runner.load_models()."""
    from real_model_runner import run_single
    out = run_single(query, model_name, use_rag, models_dict)
    return {
        "response": out["response"],
        "response_tokens": out["response_tokens"],
        "latency_seconds": out["latency_seconds"],
        "model": out["model"],
        "rag": out["rag"],
        "evidence": out.get("evidence", ""),
        "query": query,
    }

def load_queries(max_queries: int = 50, data_dir: str = None) -> List[str]:
    try:
        from paths_config import data_path
        path = data_path("queries_expanded.json") if data_dir is None else os.path.join(data_dir, "queries_expanded.json")
    except ImportError:
        path = "queries_expanded.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("queries", [])[:max_queries]
    return [
        "What is the treatment for diabetes and its associated management?",
        "How Can AI Assist in the Diagnosis of Disease?",
        "What Are the Current Guidelines for the Management of Hypertension?",
    ] * 4

def run_2x2_eval(
    queries: List[str],
    use_real_model: bool = False,
    models_dict: dict = None,
) -> List[Dict]:
    """
    Run 2x2: for each query x (SLM/LLM) x (RAG/NoRAG) produce one response and metrics.
    use_real_model: if True, call actual model generate; else mock.
    """
    results = []
    if use_real_model and models_dict:
        for q in queries:
            for model in ["slm", "llm"]:
                for rag in [False, True]:
                    resp = _real_inference(q, model, rag, models_dict)
                    resp["query"] = q
                    results.append(resp)
    else:
        for q in queries:
            for model in ["slm", "llm"]:
                for rag in [False, True]:
                    resp = _mock_inference(q, model, rag)
                    resp["query"] = q
                    results.append(resp)
    return results

def main(use_real_model: bool = None, max_queries: int = 50, data_dir: str = None, output_dir: str = None):
    if use_real_model is None:
        use_real_model = os.environ.get("EVAL_USE_REAL_MODEL", "").lower() in ("1", "true", "yes")
    try:
        from paths_config import data_path, output_path, DATA_DIR, OUTPUT_DIR
        if data_dir is None:
            data_dir = DATA_DIR
        if output_dir is None:
            output_dir = OUTPUT_DIR
    except ImportError:
        data_dir = data_dir or "."
        output_dir = output_dir or "."
        def data_path(p): return os.path.join(data_dir, p)
        def output_path(p): return os.path.join(output_dir, p)
    print("Step 1: Load expanded queries")
    queries = load_queries(max_queries, data_dir=data_dir)
    print(f"  Loaded {len(queries)} queries")

    models_dict = None
    if use_real_model:
        print("  Loading real models (HF token from HF_TOKEN or HUGGING_FACE_HUB_TOKEN)...")
        try:
            from real_model_runner import load_models
            models_dict = load_models(use_4bit=True)
            print("  Models loaded.")
        except Exception as e:
            print(f"  Real model load failed: {e}. Falling back to mock.")
            use_real_model = False

    print("Step 2: Run 2x2 evaluation (" + ("real" if use_real_model else "mock") + " inference)")
    raw = run_2x2_eval(queries, use_real_model=use_real_model, models_dict=models_dict)
    print(f"  Total runs: {len(raw)}")

    print("Step 3: Quality (F1) vs references")
    try:
        from eval_quality_metrics import load_references, evaluate_with_references
        refs_path = os.path.join(data_dir, "references.json")
        refs = load_references(refs_path)
        # Build predictions list (one per query; use first model condition for demo)
        preds_by_q = defaultdict(list)
        for r in raw:
            preds_by_q[r["query"]].append(r)
        preds = []
        for q in queries[:10]:
            rs = preds_by_q.get(q, [])
            if rs:
                preds.append({"query": q, "response": rs[0]["response"]})
        qual = evaluate_with_references(preds, refs, query_key="query", pred_key="response")
        print(f"  Mean F1: {qual.get('mean_f1')}; n with ref: {qual.get('n_with_ref')}")
    except Exception as e:
        qual = {"error": str(e)}
        print("  Error:", e)

    print("Step 4: Safety (claim verification) aggregate")
    try:
        from eval_safety_verification import run_safety_eval
        with_ev = [{"response": r["response"], "evidence": r.get("evidence", "")} for r in raw[:20]]
        safety = run_safety_eval(with_ev)
        print(f"  Aggregate: {safety['aggregate']}")
    except Exception as e:
        safety = {"error": str(e)}
        print("  Error:", e)

    print("Step 5: Energy and carbon")
    try:
        from eval_energy_carbon import aggregate_energy_carbon
        energy = aggregate_energy_carbon(raw, token_key="response_tokens")
        print(f"  Total kWh: {energy['total_kwh']}; mean per query: {energy['mean_kwh_per_query']}")
    except Exception as e:
        energy = {"error": str(e)}
        print("  Error:", e)

    print("Step 6: Routing evaluation (train/test split)")
    try:
        from eval_routing import evaluate_routing
        rout = evaluate_routing(queries, test_frac=0.3, seed=42)
        print(f"  Test accuracy: {rout['test_accuracy']}; n_test: {rout['n_test']}")
    except Exception as e:
        rout = {"error": str(e)}
        print("  Error:", e)

    out = {
        "n_queries": len(queries),
        "n_runs": len(raw),
        "quality": qual,
        "safety": safety,
        "energy": energy,
        "routing": rout,
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Step 7: Wrote eval_results.json ->", results_path)
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Full eval: quality, safety, energy, routing.")
    p.add_argument("--real", action="store_true", help="Use real HF models (requires HF_TOKEN)")
    p.add_argument("--max-queries", type=int, default=50, help="Max queries (default 50)")
    args = p.parse_args()
    main(use_real_model=args.real, max_queries=args.max_queries)
