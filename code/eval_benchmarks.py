"""
Open-source HF benchmarks only (no local dataset files required).

**Single-file Kaggle:** Paste or upload only ``eval_benchmarks.py``. On first import it
base64-decodes and extracts ``real_model_runner.py``, ``eval_quality_metrics.py``, and
``measurement_config.py`` into ``/kaggle/working`` (or ``.`` locally if helpers are missing).
Install pip deps, set ``HF_TOKEN``; no companion ``.py`` files or zip dataset are required.

- **MedQA (USMLE-style):** ``nnilayy/medqa-usmle`` (validation).
- **MMLU-Med:** ``cais/mmlu`` — six medical subjects (anatomy, clinical_knowledge,
  college_medicine, medical_genetics, professional_medicine, virology), ``test`` split.
- **PubMedQA:** ``pubmed_qa`` / ``pqa_labeled`` — **train** split (1k labeled; no ``validation`` split).
  Rows use real **question**, **context** (abstract), **long_answer**, **final_decision** (yes/no/maybe).
  **Do not use ``--mock``** when PubMedQA is in the run (real free-text answers only).
- **MedMCQA (optional):** ``medmcqa`` — use ``--benchmark medmcqa``; not in default ``all``.

Default ``--benchmark all`` runs **MedQA + MMLU-Med + PubMedQA** (pick **either** MedMCQA **or**
MMLU-Med as your second MCQ track; here MMLU-Med is the default pair with MedQA).

**Step 1:** Subset with ``--max_items 750`` and ``--seed 42``; cache predictions via
``--save_predictions /kaggle/working/my_run``.

**Output:** After a successful run, aggregated metrics are written to ``benchmark_results_all.json``
in the working directory: ``/kaggle/working/benchmark_results_all.json`` on Kaggle, or
``./benchmark_results_all.json`` when running locally. Use ``--out_json /path/to/file.json`` to
choose a different path.

Four configs: SLM_NoRAG, SLM_RAG, LLM_NoRAG, LLM_RAG. Each SLM/LLM is loaded **once**
for all benchmarks (not per dataset). RAG index is **prewarmed** before inference. MCQ and
PubMedQA label prompts use **greedy short decoding** (same task, much less generation time).

**RAG (optional):** Build ``index.faiss`` + ``chunks.jsonl`` (same embedding model at
build and query time) via ``python build_rag_index.py --input chunks.jsonl --out_dir /path``.
On Kaggle, writing to ``/kaggle/working/rag_index`` is auto-detected if you omit
``--rag_index_dir``. Otherwise pass ``--rag_index_dir /path`` (or ``--rag_faiss`` /
``--rag_chunks`` separately), ``--rag_top_k``, ``--rag_context_max_chars``,
``--rag_use_mock`` to force mock evidence, and ``--rag_embed_model`` to match the index.
Quantization is controlled by ``--no_4bit`` (model load only; unrelated to FAISS).

    pip install -q datasets rouge-score accelerate
    python eval_benchmarks.py --max_items 750 --seed 42 --save_predictions /kaggle/working/bench_v1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import base64
import io
import zipfile

# Single-file bundle: measurement_config, eval_quality_metrics, real_model_runner (extract on demand).
_BUNDLED_PY_ZIP_B64 = """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"""


def _unpack_bundled_helpers(work_dir: str) -> None:
    raw = base64.b64decode(_BUNDLED_PY_ZIP_B64)
    os.makedirs(work_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        zf.extractall(work_dir)
    print(f"Unpacked bundled helpers into {work_dir}", flush=True)



def _work_dir() -> str:
    return "/kaggle/working" if os.path.exists("/kaggle") else "."


def _default_out_json() -> str:
    return os.path.join(_work_dir(), "benchmark_results_all.json")


def _kaggle_input_code_dirs() -> List[str]:
    """Paths under /kaggle/input where a dataset might ship the four .py modules."""
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    out: List[str] = []
    try:
        for name in sorted(os.listdir(base)):
            root = os.path.join(base, name)
            if not os.path.isdir(root):
                continue
            out.append(root)
            try:
                for sub in os.listdir(root):
                    sp = os.path.join(root, sub)
                    if os.path.isdir(sp):
                        out.append(sp)
            except OSError:
                pass
    except OSError:
        pass
    return out


def _code_roots() -> List[str]:
    roots: List[str] = []
    cd = (os.environ.get("CODE_DIR") or os.environ.get("GREEN_PAPER_CODE") or "").strip()
    if cd:
        roots.append(cd)
    if os.path.isdir("/kaggle/working"):
        roots.append("/kaggle/working")
    roots.extend(_kaggle_input_code_dirs())
    try:
        h = os.path.dirname(os.path.abspath(__file__))
        if h:
            roots.append(h)
    except NameError:
        pass
    roots.append(os.getcwd())
    seen: set[str] = set()
    out: List[str] = []
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _setup_runner_path() -> str:
    work = _work_dir()
    on_kaggle = os.path.isdir("/kaggle/working")

    def _try_existing() -> Optional[str]:
        for root in _code_roots():
            if os.path.isfile(os.path.join(root, "real_model_runner.py")):
                if root not in sys.path:
                    sys.path.insert(0, root)
                return root
        return None

    if on_kaggle:
        _unpack_bundled_helpers(work)
        _drop_runner_caches()
        if work not in sys.path:
            sys.path.insert(0, work)
        if os.path.isfile(os.path.join(work, "real_model_runner.py")):
            return work
        hit = _try_existing()
        if hit:
            return hit
        raise ImportError("Bundled extract failed: real_model_runner.py missing after unpack.")

    hit = _try_existing()
    if hit:
        return hit
    _unpack_bundled_helpers(work)
    _drop_runner_caches()
    if work not in sys.path:
        sys.path.insert(0, work)
    if os.path.isfile(os.path.join(work, "real_model_runner.py")):
        return work
    raise ImportError(
        "real_model_runner.py not found after extracting bundled helpers. "
        "Run from a writable directory or set CODE_DIR to a folder with the four .py files."
    )


def _drop_runner_caches() -> None:
    m = sys.modules.get("real_model_runner")
    if m is not None and hasattr(m, "_clear_rag_cache"):
        try:
            m._clear_rag_cache()
        except Exception:
            pass
    for mod in ("real_model_runner", "eval_quality_metrics", "measurement_config"):
        sys.modules.pop(mod, None)


def _run_out(raw_or_obj: Any) -> Tuple[str, str, str]:
    """Normalize run_fn return: (response_text, retrieved_context, rag_source)."""
    if isinstance(raw_or_obj, dict):
        return (
            str(raw_or_obj.get("response", "")),
            str(raw_or_obj.get("retrieved_context") or raw_or_obj.get("evidence") or ""),
            str(raw_or_obj.get("rag_source") or ""),
        )
    return str(raw_or_obj), "", ""


def _nonempty_env(*keys: str) -> bool:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip():
            return True
    return False


def _ensure_hf_token() -> None:
    if _nonempty_env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        return

    try:
        from huggingface_hub import get_token

        t = get_token()
        if t and str(t).strip():
            os.environ.setdefault("HF_TOKEN", str(t).strip())
            print("Loaded HF token from huggingface_hub cache (get_token).", flush=True)
            return
    except Exception:
        pass

    try:
        from kaggle_secrets import UserSecretsClient

        c = UserSecretsClient()
        for name in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "hf_token",
            "HUGGINGFACE_HUB_TOKEN",
            "huggingface_token",
            "HF",
        ):
            try:
                v = c.get_secret(name)
                if v and str(v).strip():
                    os.environ["HF_TOKEN"] = str(v).strip()
                    print(f"Loaded HF_TOKEN from Kaggle Secrets (label: {name}).", flush=True)
                    return
            except Exception:
                continue
    except Exception:
        pass

    for p in ("/kaggle/secret/hf_token", "/kaggle/secret/HF_TOKEN"):
        if os.path.isfile(p):
            os.environ["HF_TOKEN"] = open(p, encoding="utf-8").read().strip()
            print(f"Loaded HF_TOKEN from {p}.", flush=True)
            return

    on_kaggle = os.path.isdir("/kaggle") and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    hint = (
        "HF token missing (gated models such as Llama need it).\n\n"
        "On Kaggle:\n"
        "  1) Create a secret at https://www.kaggle.com/settings (Secrets) with label HF_TOKEN "
        "(or HUGGING_FACE_HUB_TOKEN) and your Hugging Face read token.\n"
        "  2) In this notebook: Add-ons → Secrets (or the key icon) and turn ON access for that secret "
        "(without this, get_secret never sees it).\n"
        "  3) Or in a cell above: os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n\n"
        "Smoke test without a token: add --mock (MCQ benchmarks only; not for PubMedQA).\n"
    )
    if not on_kaggle:
        hint += "\nLocally: export HF_TOKEN=... or huggingface-cli login.\n"
    raise RuntimeError(hint)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise ValueError(f"Bad JSON shape: {path}")


def _trim_or_sample(
    items: List[Dict[str, Any]], max_items: int, seed: Optional[int]
) -> List[Dict[str, Any]]:
    """If max_items <= 0, use full list. Else take first max_items, or random sample if seed set."""
    if max_items <= 0 or max_items >= len(items):
        return items
    if seed is None:
        return items[:max_items]
    rng = random.Random(int(seed))
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:max_items]]


def _load_medqa(split: str = "validation") -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("nnilayy/medqa-usmle", split=split)
    items: List[Dict[str, Any]] = []
    for row in ds:
        s1 = str(row.get("sent1") or "").strip()
        s2 = str(row.get("sent2") or "").strip()
        q = f"{s1}\n{s2}".strip() if s2 else s1
        choices = [str(row.get(f"ending{i}", "") or "") for i in range(4)]
        lbl = row.get("label")
        try:
            i = int(lbl)
        except (TypeError, ValueError):
            i = -1
        ans = chr(ord("A") + i) if 0 <= i <= 3 else str(lbl)
        items.append({"id": str(row.get("id", "")), "question": q, "choices": choices, "answer": ans})
    return items


def _load_medmcqa(split: str = "validation") -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("medmcqa", split=split)
    items: List[Dict[str, Any]] = []
    for row in ds:
        items.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question") or "",
                "choices": [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")],
                "answer": str(row.get("cop") or row.get("answer") or ""),
            }
        )
    return items


# HF ``pqa_labeled`` only exposes ``train`` (1000 QA-RL examples with yes/no/maybe labels).
PUBMEDQA_LABELED_SPLIT = "train"


def _normalize_pubmed_label(s: str) -> str:
    """Map text to a single label yes | no | maybe; empty if ambiguous/unknown."""
    t = (s or "").strip().lower()
    if not t:
        return ""
    m = re.match(r"^(yes|no|maybe)\b", t)
    if m:
        return m.group(1)
    if re.search(r"\bmaybe\b|\bunsure\b|insufficient|not enough evidence", t):
        return "maybe"
    has_yes = bool(re.search(r"\byes\b", t))
    has_no = bool(re.search(r"\bno\b", t))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return ""


def _parse_pubmed_model_answer(raw: str) -> str:
    """Map model output to a single label for accuracy (real eval, not mock)."""
    return _normalize_pubmed_label(raw)


def _flatten_pubmed_context(context_field: str) -> str:
    """
    PubMedQA stores ``context`` as a stringified dict with ``contexts`` (abstract snippets).
    Join those strings for real prompts; fall back to raw text if parsing fails.
    """
    raw = (context_field or "").strip()
    if not raw:
        return ""
    if raw.startswith("{") and "contexts" in raw:
        try:
            import ast

            d = ast.literal_eval(raw)
            ctxs = d.get("contexts")
            if isinstance(ctxs, list) and ctxs:
                return "\n\n".join(str(c).strip() for c in ctxs if c)
        except (SyntaxError, ValueError, TypeError):
            pass
    return raw


def _pubmedqa_user_prompt(question: str, context: str) -> str:
    """Real PubMedQA-style prompt: abstract context + question (no fabricated demo text)."""
    q = (question or "").strip()
    ctx = (context or "").strip()
    if ctx:
        return (
            "Read the biomedical abstract below, then answer the question with exactly one word: "
            "yes, no, or maybe.\n\n"
            f"Abstract:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            "Answer (one word only):"
        )
    return (
        f"{q}\n\nAnswer with exactly one word: yes, no, or maybe."
    )


def _load_pubmedqa(split: str = PUBMEDQA_LABELED_SPLIT) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    items: List[Dict[str, Any]] = []
    for row in ds:
        dec = _normalize_pubmed_label(str(row.get("final_decision") or row.get("answer") or ""))
        if not dec:
            continue
        pid = row.get("pubid", row.get("id", ""))
        ctx_raw = str(row.get("context") or "").strip()
        ctx_flat = _flatten_pubmed_context(ctx_raw)
        items.append(
            {
                "id": str(pid),
                "question": str(row.get("question") or "").strip(),
                "context": ctx_flat,
                "context_raw_stored": ctx_raw,
                "long_answer": str(row.get("long_answer") or "").strip(),
                "answer": dec,
            }
        )
    return items


# Medical-related MMLU subjects (``cais/mmlu``), standard ``test`` split for reporting.
MMLU_MED_SUBJECTS: Tuple[str, ...] = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
)


def _mmlu_answer_letter(row: Dict[str, Any]) -> str:
    a = row["answer"]
    if isinstance(a, str):
        idx = ord(a.strip().upper()) - ord("A")
    else:
        idx = int(a)
    if 0 <= idx <= 3:
        return chr(ord("A") + idx)
    return str(a)


def _load_mmlu_med(split: str = "test") -> List[Dict[str, Any]]:
    """MMLU medical slice: multiple subjects from ``cais/mmlu`` (4-way MCQ)."""
    from datasets import load_dataset  # type: ignore

    items: List[Dict[str, Any]] = []
    for subj in MMLU_MED_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split=split)
        for i, row in enumerate(ds):
            choices = [str(c) for c in row["choices"]]
            items.append(
                {
                    "id": f"{subj}_{i}",
                    "question": str(row.get("question", "")).strip(),
                    "choices": choices,
                    "answer": _mmlu_answer_letter(row),
                }
            )
    return items


def _bootstrap_ci(values: List[float], alpha: float = 0.05, iters: int = 2000) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    samples: List[float] = []
    for _ in range(iters):
        samples.append(sum(values[random.randint(0, n - 1)] for _ in range(n)) / n)
    samples.sort()
    lo = int((alpha / 2) * iters)
    hi = int((1 - alpha / 2) * iters)
    return mean, samples[lo], samples[min(hi, iters - 1)]


def _accuracy(preds: List[Any], golds: List[Any]) -> Tuple[float, float, float]:
    return _bootstrap_ci([1.0 if p == g else 0.0 for p, g in zip(preds, golds)])


def _token_f1(pred: str, ref: str) -> float:
    from eval_quality_metrics import compute_f1

    return compute_f1(pred, ref)["f1"]


def _rouge_l(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return _bootstrap_ci(scores)


def _normalize_mcq_gold(gold: str, num_choices: int) -> str:
    g = str(gold).strip().upper()
    if len(g) == 1 and "A" <= g <= "Z" and ord(g) - ord("A") < num_choices:
        return g
    if g.isdigit():
        i = int(g)
        if 0 <= i < num_choices:
            return chr(ord("A") + i)
    return g[:1] if g else ""


def _extract_mcq_letter(raw: str, num_choices: int) -> str:
    if not raw:
        return ""
    u = raw.upper()
    last = chr(ord("A") + max(0, num_choices - 1))
    m = re.search(rf"\b([A-{last}])\b", u)
    if m:
        return m.group(1)
    m2 = re.search(r"ANSWER\s*[:.]?\s*([A-Z])", u)
    if m2 and "A" <= m2.group(1) <= last:
        return m2.group(1)
    for ch in u:
        if "A" <= ch <= last:
            return ch
    return ""


CONFIGS: List[Tuple[str, str, bool]] = [
    ("SLM_NoRAG", "slm", False),
    ("SLM_RAG", "slm", True),
    ("LLM_NoRAG", "llm", False),
    ("LLM_RAG", "llm", True),
]


def _evaluate_mcq_for_model(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run NoRAG + RAG configs for one model (keeps a single GPU model loaded)."""
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for cfg_name, mk, use_rag in CONFIGS:
        if mk != model_key:
            continue
        preds: List[str] = []
        golds: List[str] = []
        for item in items:
            choices = [c for c in item["choices"] if c is not None]
            n = len(choices)
            gold = _normalize_mcq_gold(str(item["answer"]), n)
            prompt = (
                item["question"]
                + "\n\nOptions:\n"
                + "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
                + "\n\nReply with only the single letter of the best option."
            )
            out = run_fn(model_key, prompt, use_rag)
            raw, rctx, rsrc = _run_out(out)
            pred = _extract_mcq_letter(raw, n)
            preds.append(pred)
            golds.append(gold)
            rows.append(
                {
                    "benchmark": benchmark,
                    "question_id": str(item.get("id", "")),
                    "question": item["question"],
                    "reference_answer": gold,
                    "model_name": cfg_name,
                    "model_key": model_key,
                    "rag_flag": use_rag,
                    "model_answer": pred,
                    "raw_response": raw,
                    "retrieved_context": rctx if use_rag else "",
                    "rag_source": rsrc if use_rag else "",
                    "parsed_prediction": pred,
                    "mcq_correct": pred == gold,
                    "choices_json": json.dumps(choices, ensure_ascii=False),
                    "token_f1": "",
                }
            )
        mean, lo, hi = _accuracy(preds, golds)
        results[cfg_name] = {"metric": "accuracy", "mean": mean, "ci_lower": lo, "ci_upper": hi, "n": len(items)}
    return results, rows


def _evaluate_mcq(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for model_key in ("slm", "llm"):
        r, rw = _evaluate_mcq_for_model(items, run_fn, benchmark, model_key)
        results.update(r)
        rows.extend(rw)
    return results, rows


def _evaluate_free_text_for_model(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    use_pubmed_prompt = benchmark == "pubmedqa"

    for cfg_name, mk, use_rag in CONFIGS:
        if mk != model_key:
            continue
        preds: List[str] = []
        refs: List[str] = []
        f1s: List[float] = []
        label_hits: List[float] = []

        for item in items:
            q = item["question"]
            ref = str(item["answer"] or "").strip().lower()
            if use_pubmed_prompt:
                prompt = _pubmedqa_user_prompt(q, str(item.get("context") or ""))
            else:
                prompt = f"{q}\n\nAnswer briefly: yes, no, or maybe only."

            out = run_fn(model_key, prompt, use_rag)
            raw, rctx, rsrc = _run_out(out)
            preds.append(raw)
            refs.append(ref)
            f1s.append(_token_f1(raw, ref))

            parsed = _parse_pubmed_model_answer(raw) if use_pubmed_prompt else ""
            if use_pubmed_prompt and ref in ("yes", "no", "maybe"):
                label_hits.append(1.0 if parsed == ref else 0.0)
            else:
                label_hits.append(float("nan"))

            rows.append(
                {
                    "benchmark": benchmark,
                    "question_id": str(item.get("id", "")),
                    "question": q,
                    "reference_answer": ref,
                    "context_chars": len(str(item.get("context") or "")),
                    "model_name": cfg_name,
                    "model_key": model_key,
                    "rag_flag": use_rag,
                    "model_answer": raw,
                    "raw_response": raw,
                    "retrieved_context": rctx if use_rag else "",
                    "rag_source": rsrc if use_rag else "",
                    "parsed_prediction": parsed,
                    "label_correct": (parsed == ref) if use_pubmed_prompt and ref and parsed else "",
                    "mcq_correct": "",
                    "choices_json": "",
                    "token_f1": f1s[-1],
                }
            )

        rm, rl, rh = _rouge_l(preds, refs)
        fm, fl, fh = _bootstrap_ci(f1s)
        out: Dict[str, Any] = {
            "metric": "free_text",
            "rougeL_mean": rm,
            "rougeL_ci_lower": rl,
            "rougeL_ci_upper": rh,
            "f1_mean": fm,
            "f1_ci_lower": fl,
            "f1_ci_upper": fh,
            "n": len(items),
        }
        if use_pubmed_prompt:
            clean_hits = [x for x in label_hits if x == x]  # drop NaN
            if clean_hits:
                lm, ll, lh = _bootstrap_ci(clean_hits)
                out["pubmedqa_label_accuracy_mean"] = lm
                out["pubmedqa_label_accuracy_ci_lower"] = ll
                out["pubmedqa_label_accuracy_ci_upper"] = lh
        results[cfg_name] = out
    return results, rows


def _evaluate_free_text(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for model_key in ("slm", "llm"):
        r, rw = _evaluate_free_text_for_model(items, run_fn, benchmark, model_key)
        results.update(r)
        rows.extend(rw)
    return results, rows


def _prewarm_rag_before_benchmarks() -> None:
    if not any(use_rag for _n, _k, use_rag in CONFIGS):
        return
    try:
        _setup_runner_path()
        import real_model_runner as rmr

        if rmr.prewarm_rag_index():
            print("RAG: index prewarmed (FAISS + embedder cached for all questions).", flush=True)
    except Exception as ex:
        print(f"RAG prewarm skipped: {ex}", flush=True)


def _load_items(
    name: str, data_path: str, max_items: int, subset_seed: Optional[int]
) -> List[Dict[str, Any]]:
    if data_path:
        items = _load_jsonl(data_path) if data_path.endswith(".jsonl") else _load_json(data_path)
    elif name == "medqa":
        items = _load_medqa("validation")
    elif name == "medmcqa":
        items = _load_medmcqa("validation")
    elif name == "pubmedqa":
        items = _load_pubmedqa(PUBMEDQA_LABELED_SPLIT)
    elif name == "mmlu_med":
        items = _load_mmlu_med("test")
    else:
        raise ValueError(f"'{name}' needs --data_path")
    return _trim_or_sample(items, max_items, subset_seed)


def extract_open_benchmark_datasets(out_dir: str, secondary_mcq: str = "mmlu_med") -> Dict[str, str]:
    """
    Download open HF benchmarks and write JSONL (real rows, no models, no mock).
    - ``medqa_usmle.jsonl`` — USMLE-style MedQA.
    - ``mmlu_med.jsonl`` *or* ``medmcqa.jsonl`` — second MCQ track.
    - ``pubmedqa_pqa_labeled.jsonl`` — question, context, long_answer, answer (yes/no/maybe).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    medqa = _load_medqa("validation")
    p_mq = os.path.join(out_dir, "medqa_usmle.jsonl")
    with open(p_mq, "w", encoding="utf-8") as f:
        for row in medqa:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["medqa_usmle"] = p_mq

    if secondary_mcq == "mmlu_med":
        sec_rows = _load_mmlu_med("test")
        p_sec = os.path.join(out_dir, "mmlu_med.jsonl")
    elif secondary_mcq == "medmcqa":
        sec_rows = _load_medmcqa("validation")
        p_sec = os.path.join(out_dir, "medmcqa.jsonl")
    else:
        raise ValueError(f"secondary_mcq must be mmlu_med or medmcqa, got {secondary_mcq!r}")
    with open(p_sec, "w", encoding="utf-8") as f:
        for row in sec_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths[secondary_mcq] = p_sec

    pub = _load_pubmedqa(PUBMEDQA_LABELED_SPLIT)
    p_pb = os.path.join(out_dir, "pubmedqa_pqa_labeled.jsonl")
    with open(p_pb, "w", encoding="utf-8") as f:
        for row in pub:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["pubmedqa_pqa_labeled"] = p_pb

    print("Extracted datasets (JSONL):", flush=True)
    for k, v in paths.items():
        print(f"  {k}: {v}", flush=True)
    return paths


def _write_prediction_artifacts(
    prefix: str,
    all_rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    """Write ``{prefix}_predictions.json`` and ``{prefix}_predictions.csv``."""
    prefix = prefix.rstrip(".json").rstrip(".csv")
    if prefix.endswith("_predictions"):
        base = prefix
    else:
        base = f"{prefix}_predictions"
    jpath = f"{base}.json"
    cpath = f"{base}.csv"
    jdir = os.path.dirname(os.path.abspath(jpath))
    if jdir:
        os.makedirs(jdir, exist_ok=True)
    payload = {"meta": meta, "n_rows": len(all_rows), "rows": all_rows}
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if not all_rows:
        print(f"Wrote {jpath} (0 rows); skipped CSV", flush=True)
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {jpath} and {cpath} ({len(all_rows)} rows)", flush=True)


def run_all_benchmarks(
    benchmarks: List[str],
    max_items: int,
    use_4bit: bool,
    data_paths: Dict[str, str],
    out_json: str,
    mock: bool = False,
    subset_seed: Optional[int] = None,
    save_predictions_prefix: str = "",
    rag_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    for d in (_work_dir(),):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here and here not in sys.path:
            sys.path.insert(0, here)
    except NameError:
        pass

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    if mock and "pubmedqa" in benchmarks:
        raise RuntimeError(
            "PubMedQA must use real model outputs (full abstract + yes/no/maybe). "
            "Re-run without --mock, or drop pubmedqa from this run."
        )

    if mock:
        print("MOCK: no GPU models (MCQ-only fake answers).", flush=True)

        def run_fn(model_key: str, question: str, use_rag: bool) -> str:
            return "A" if "OPTIONS" in question.upper() else "yes"

    else:
        _setup_runner_path()
        _drop_runner_caches()
        _ensure_hf_token()
        import torch
        from real_model_runner import run_single
        try:
            from real_model_runner import load_one_model  # type: ignore
        except Exception:
            # Compatibility: older/partial `real_model_runner.py` may not define `load_one_model`.
            # Fall back to `load_models` and select the requested model.
            from real_model_runner import load_models  # type: ignore

            def load_one_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any]:
                models_all = load_models(use_4bit=use_4bit)
                return models_all[model_key]

        print(
            "Models load one-at-a-time per SLM/LLM block (lower peak VRAM; helps Kaggle OOM/kernel death).",
            flush=True,
        )
        models: Dict[str, Tuple[Any, Any]] = {}
        _active_model_key: Optional[str] = None

        def _ensure_model_loaded(model_key: str) -> None:
            nonlocal _active_model_key
            if _active_model_key == model_key and model_key in models:
                return
            models.clear()
            _active_model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Loading {model_key.upper()}...", flush=True)
            assert model_key in ("slm", "llm")
            m, tok = load_one_model(model_key, use_4bit=use_4bit)
            models[model_key] = (m, tok)
            _active_model_key = model_key

        def run_fn(model_key: str, question: str, use_rag: bool) -> Any:
            _ensure_model_loaded(model_key)
            return run_single(question, model_key=model_key, use_rag=use_rag, models_dict=models)

    agg: Dict[str, Any] = {
        "max_items_per_benchmark": max_items,
        "subset_seed": subset_seed,
        "use_4bit": use_4bit,
        "mock": mock,
        "benchmarks": {},
    }
    if rag_config:
        agg["rag_config"] = rag_config

    prediction_rows: List[Dict[str, Any]] = []
    mcq_benches = {"medqa", "medmcqa", "mmlu_med", "custom_mcq"}
    bench_items: Dict[str, List[Dict[str, Any]]] = {}
    bench_is_mcq: Dict[str, bool] = {}

    for name in benchmarks:
        dp = data_paths.get(name, "")
        print(f"\n=== {name} === (loading)", flush=True)
        items = _load_items(name, dp, max_items, subset_seed)
        bench_items[name] = items
        bench_is_mcq[name] = name in mcq_benches
        print(f"  n={len(items)}", flush=True)
        if not items:
            agg["benchmarks"][name] = {"error": "no_items"}

    if not mock:
        _prewarm_rag_before_benchmarks()

    # Load each GPU model once for all benchmarks (2 loads vs 2×#benchmarks).
    model_keys = ("slm", "llm")
    for model_key in model_keys:
        print(f"\n--- Model block: {model_key.upper()} (all benchmarks) ---", flush=True)
        for name in benchmarks:
            items = bench_items.get(name) or []
            if not items:
            continue
            print(f"\n=== {name} === [{model_key.upper()}]", flush=True)
        try:
                if bench_is_mcq[name]:
                    res, rows = _evaluate_mcq_for_model(items, run_fn, name, model_key)
            else:
                    res, rows = _evaluate_free_text_for_model(items, run_fn, name, model_key)
            prediction_rows.extend(rows)
                prev = agg["benchmarks"].get(name)
                if prev and "results" in prev:
                    prev["results"].update(res)
                else:
                    agg["benchmarks"][name] = {"n_items": len(items), "results": dict(res)}
        except Exception as ex:
            agg["benchmarks"][name] = {"error": str(ex)}
            print(f"  ERROR: {ex}", flush=True)

    if save_predictions_prefix.strip():
        _write_prediction_artifacts(
            save_predictions_prefix.strip(),
            prediction_rows,
            {
                "benchmarks": benchmarks,
                "max_items": max_items,
                "subset_seed": subset_seed,
                "mock": mock,
                "configs": [c[0] for c in CONFIGS],
            },
        )

    out_dir = os.path.dirname(os.path.abspath(out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"\nWrote {out_json}", flush=True)
    return agg


def _apply_rag_env_from_args(ns: Any) -> Dict[str, str]:
    """Apply --rag_* CLI flags to the environment; return a snapshot for results JSON."""
    if ns.rag_index_dir.strip():
        os.environ["RAG_INDEX_DIR"] = os.path.abspath(ns.rag_index_dir.strip())
    if ns.rag_faiss.strip():
        os.environ["RAG_FAISS_INDEX"] = os.path.abspath(ns.rag_faiss.strip())
    if ns.rag_chunks.strip():
        os.environ["RAG_CHUNKS_JSONL"] = os.path.abspath(ns.rag_chunks.strip())
    os.environ["RAG_TOP_K"] = str(int(ns.rag_top_k))
    os.environ["RAG_CONTEXT_MAX_CHARS"] = str(int(ns.rag_context_max_chars))
    if ns.rag_use_mock:
        os.environ["RAG_FORCE_MOCK"] = "1"
    else:
        os.environ.pop("RAG_FORCE_MOCK", None)
    if ns.rag_embed_model.strip():
        os.environ["RAG_EMBED_MODEL"] = ns.rag_embed_model.strip()
    else:
        os.environ.pop("RAG_EMBED_MODEL", None)
    keys = (
        "RAG_INDEX_DIR",
        "RAG_FAISS_INDEX",
        "RAG_CHUNKS_JSONL",
        "RAG_TOP_K",
        "RAG_CONTEXT_MAX_CHARS",
        "RAG_EMBED_MODEL",
        "RAG_FORCE_MOCK",
    )
    return {k: os.environ.get(k, "") for k in keys}


def main() -> None:
    p = argparse.ArgumentParser(description="Medical QA benchmarks.")
    p.add_argument(
        "--benchmark",
        default="all",
        choices=[
            "all",
            "medqa",
            "mmlu_med",
            "medmcqa",
            "pubmedqa",
            "custom_mcq",
            "custom_free",
        ],
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=500,
        help="Cap per benchmark (default 500; 0 = full HF split). Use --seed for reproducible random subsets.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random subset seed (only when max_items > 0). Omit to take the first N examples in dataset order.",
    )
    p.add_argument(
        "--save_predictions",
        default="",
        help="Path prefix: writes {prefix}_predictions.json and .csv (all benchmarks, all configs).",
    )
    p.add_argument("--out_json", default="", help=f"default: {_default_out_json()}")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (ablation: less compression loss, more VRAM).")
    p.add_argument(
        "--rag_index_dir",
        default="",
        help="Folder with index.faiss + chunks.jsonl. Sets RAG_INDEX_DIR. On Kaggle, "
        "/kaggle/working/rag_index is used automatically if this is omitted and those files exist.",
    )
    p.add_argument(
        "--rag_faiss",
        default="",
        help="Path to FAISS index file (overrides dir). Sets RAG_FAISS_INDEX.",
    )
    p.add_argument(
        "--rag_chunks",
        default="",
        help="Path to chunks JSONL (same row order as index). Sets RAG_CHUNKS_JSONL.",
    )
    p.add_argument("--rag_top_k", type=int, default=5, help="Number of chunks to retrieve (default 5).")
    p.add_argument(
        "--rag_context_max_chars",
        type=int,
        default=6000,
        help="Max characters of retrieved text in the prompt (ablation: context overload).",
    )
    p.add_argument(
        "--rag_use_mock",
        action="store_true",
        help="Force mock RAG even if a FAISS index is configured (ablation baseline).",
    )
    p.add_argument(
        "--rag_embed_model",
        default="",
        help="Sentence-Transformers model id for query encoding (must match index). Default: all-MiniLM-L6-v2 in runner.",
    )
    p.add_argument("--data_path", default="", help="for custom_mcq / custom_free")
    p.add_argument(
        "--extract_dir",
        default="",
        help="If set: download MedQA + secondary MCQ + PubMedQA to this folder as JSONL, then exit (no inference).",
    )
    p.add_argument(
        "--secondary_mcq",
        default="mmlu_med",
        choices=["mmlu_med", "medmcqa"],
        help="With --extract_dir: which second MCQ JSONL to write.",
    )
    p.add_argument("--mock", action="store_true")
    args, unknown = p.parse_known_args()
    rest: List[str] = []
    i = 0
    while i < len(unknown):
        if unknown[i] == "-f" and i + 1 < len(unknown):
            i += 2
        else:
            rest.append(unknown[i])
            i += 1
    if rest:
        print("Ignoring:", rest, flush=True)

    if args.extract_dir.strip():
        extract_open_benchmark_datasets(args.extract_dir.strip(), secondary_mcq=args.secondary_mcq)
        return

    if args.benchmark == "all":
        # MedQA + MMLU-Med + PubMedQA (open HF only). Use --benchmark medmcqa for MedMCQA instead/in addition.
        benches, dmap = ["medqa", "mmlu_med", "pubmedqa"], {}
    elif args.benchmark in {"custom_mcq", "custom_free"}:
        if not args.data_path:
            print("--data_path required", file=sys.stderr)
            sys.exit(2)
        benches, dmap = [args.benchmark], {args.benchmark: args.data_path}
    else:
        benches, dmap = [args.benchmark], {}

    rag_snapshot = _apply_rag_env_from_args(args)
    run_all_benchmarks(
        benchmarks=benches,
        max_items=args.max_items,
        use_4bit=not args.no_4bit,
        data_paths=dmap,
        out_json=args.out_json or _default_out_json(),
        mock=args.mock,
        subset_seed=args.seed,
        save_predictions_prefix=args.save_predictions,
        rag_config=rag_snapshot,
    )


if __name__ == "__main__":
    main()