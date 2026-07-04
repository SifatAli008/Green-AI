#!/usr/bin/env python3
"""
Faithfulness / hallucination scoring for Table 4 (faculty review plan §5).

Backends:
  - ``local_hf``     — single-call local HF judge (**default on Kaggle**, reliable on T4)
  - ``deepeval``     — DeepEval ``FaithfulnessMetric`` + local HF (4+ calls/row)
  - ``openrouter``   — OpenRouter API

**One Kaggle cell (paste whole file):** paste this file into a cell and run.
It auto-scores faithfulness on RAG rows (18,180) by default, or **NoRAG-only** (~18,180) via
``run_faithfulness_norag_paper_run()`` for Table~4 comparability (see ``KAGGLE_NORAG_FAITHFULNESS.md``).

**Cell 0 (once):** ``HF_TOKEN`` from secrets. Faithfulness auto-installs ``transformers``, ``accelerate``,
and (for re-retrieval) ``faiss-cpu`` + ``sentence-transformers`` unless ``GP_FAITH_NO_AUTO_PIP=1``.

**Cell 0b (RAG re-retrieval):** attach **fatinshadab/rag-index** + **rag-index-gold** (folders ``rag_index/``, ``rag_index_gold/``), set env (see below).

**Cell 0c (optional):** copy ``real_model_runner.py`` to working for full hybrid RAG; otherwise
built-in FAISS re-retrieval runs when ``faiss-cpu`` + ``sentence-transformers`` are installed.

**Cell 0c — optional full runner:** copy helper modules to working::

    import shutil, os
    for name in ("real_model_runner.py", "eval_quality_metrics.py", "measurement_config.py"):
        for base in ("/kaggle/working", os.path.join(os.getcwd(), "..", "GreenPaper_Kaggle_Benchmarks")):
            src = os.path.join(base, name)
            if os.path.isfile(src):
                shutil.copy2(src, f"/kaggle/working/{name}")
                print("copied", src)
                break

Or save ``eval_benchmarks.py`` to ``/kaggle/working/`` and run once (it unpacks helpers).

**Cell 0d (env):**::

    os.environ["GP_FAITH_CONTEXT"] = "reretrieve_or_stored"
    os.environ["GP_FAITH_CONTEXT"] = "reretrieve_or_stored"
    # Dataset root or rag_index/ subfolder (auto-discovered):
    os.environ["GP_FAITH_RAG_INDEX_DIR"] = "/kaggle/input/datasets/fatinshadab/rag-index"
    os.environ["GP_FAITH_RAG_GOLD_DIR"] = "/kaggle/input/datasets/fatinshadab/rag-index-gold"
    # Resolved dirs: .../rag-index/rag_index/  and  .../rag-index-gold/rag_index_gold/

**Cell 1 — paste entire ``run_faithfulness_eval.py`` into one cell and Run.** No separate file.
After it finishes, functions are in the notebook namespace (``run_faithfulness_paper_run()``, etc.).
Imports also work: ``from run_faithfulness_eval import ...`` (module alias registered on run).

**Cell 2 — setup (no import required if Cell 1 was pasted):**

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["GP_FAITH_AUTO"] = "0"
    os.environ["GP_FAITH_CONTEXT"] = "reretrieve_or_stored"
    os.environ["GP_FAITH_RAG_INDEX_DIR"] = "/kaggle/input/datasets/fatinshadab/rag-index/rag_index"
    os.environ["GP_FAITH_RAG_GOLD_DIR"] = "/kaggle/input/datasets/fatinshadab/rag-index-gold/rag_index_gold"
    print("primary:", _resolve_faith_rag_primary_dir())
    print("gold:", _resolve_faith_rag_gold_dir())

**Cell 3 — RAG (already done locally):** ``run_faithfulness_paper_run(rag_only=True)``

**Cell 3 — NoRAG only (Table~4 gap-fill on Kaggle):**::

    run_faithfulness_norag_paper_run(
        prior_rag_dirs="/kaggle/input/YOUR_RAG_FAITH_DATASET/fathfullness/rerun",
    )

**sifatali008 account — resume from batch 11** (batches 0–10 on another account; attach as dataset)::

    run_faithfulness_sifatali_paper_resume(
        start_batch=11,
        prior_batch_dirs="/kaggle/input/YOUR_DATASET/paper_batches",
    )

**fahim220 account — resume from batch 16** (batches 0–15 done; rows 7,501–18,180)::

    run_faithfulness_fahim220_paper_resume(
        start_batch=16,
        prior_batch_dirs="/kaggle/input/YOUR_DATASET/paper_batches",
    )

**fatinshadab account (default)** — attach three datasets; index files live under ``rag_index/`` and ``rag_index_gold/``::

    Predictions: .../benchmark-results-all-predictions-combined/benchmark_results_all_predictions_combined.csv
    RAG primary: .../rag-index/rag_index/  (chunks.jsonl, index.faiss, chunks_paper_556.jsonl, ...)
    RAG gold:    .../rag-index-gold/rag_index_gold/  (chunks.jsonl, index.faiss, pubmedqa_gold_manifest.json)

    run_faithfulness_paper_run()  # full paper run from batch 1
    # or resume:
    run_faithfulness_fatinshadab_paper_resume(start_batch=11, prior_batch_dirs="...")

**ummesalmahabiba account** — ``rag_index/`` + ``rag_index_gold/`` under attached datasets::

    RAG primary: .../rag-index/rag_index/
    RAG gold:    .../rag-index-gold/rag_index_gold/

    run_faithfulness_ummesalmahabiba_paper_resume(start_batch=36, prior_batch_dirs="...")

    Gap-fill (last 5 batches of missing list, 2,500 rows)::

        run_faithfulness_ummesalmahabiba_gap_fill()

Default predictions CSV on Kaggle (attach **fatinshadab/benchmark-results-all-predictions-combined**):

    /kaggle/input/datasets/fatinshadab/benchmark-results-all-predictions-combined/benchmark_results_all_predictions_combined.csv

Default on Kaggle: **all 18,180** ``rag_flag=True`` rows (~37 batches of 500). Skips rows already present in prior batch CSVs when ``GP_FAITH_ONLY_MISSING=1``.

Set ``GP_FAITH_PRIOR_BATCH_DIRS`` if prior batch CSVs live in an attached dataset (see example logs below).

**Expected log (resume from batch 31):**::

    [faithfulness] Starting at batch 31/37 (skip batches 1-30; rows 1-15000 assumed done; this run rows 15001-18180)
    [faithfulness] Coverage: prior_csv=0 keys on disk | GP_FAITH_START_BATCH=11 → rows 1–5000 treated as done | effective 5000/18180 | pending_this_run ~13180 | next batch 11/37
    [faithfulness] Processing batch 11/37 (...; 5000/18180 effective done, 0 from prior CSVs) -> ...batch_0010.csv
    [faithfulness] Progress: 5500/18180 RAG rows; ...

**Fresh full run:** set ``GP_FAITH_START_BATCH=1``. Default on Kaggle is **batch 31/37** (rows **15,001-18,180**, 7 batches).

Re-run the same cell until all batches finish. Outputs under
``/kaggle/working/Faithfulness/``.

Set ``GP_FAITH_AUTO=0`` to load functions only without auto-run, then call::

    run_faithfulness("", batch_size=500, rag_only=True)

Env: ``GP_FAITHFULNESS_MODEL``, ``GP_FAITH_BATCH_SIZE``, ``GP_FAITH_MAX_CTX_CHARS``,
``GP_FAITH_MCQ_ONLY`` (default 1: score selected MCQ option only),
``GP_FAITH_STRIP_PROMPT`` (default 1: judge evidence blocks only),
``GP_FAITH_CONTEXT`` (``reretrieve_or_stored`` | ``reretrieve`` | ``gold`` | ``stored``),
``GP_FAITH_RAG_INDEX_DIR`` / ``GP_FAITH_RAG_GOLD_DIR`` (default: repo ``kaggle_working/rag_index*``),
``GP_FAITH_ALLOW_STORED_FALLBACK`` (default 0: abort if re-retrieval helpers missing),
``GP_FAITH_ROW_LIMIT``, ``GP_FAITH_ROW_OFFSET`` / ``GP_FAITH_LAST_ROWS`` (slice after RAG filter; default 0),
``GP_FAITH_ONLY_MISSING`` (default 1: skip keys already in batch CSVs under output / prior dirs),
``GP_FAITH_PRIOR_BATCH_DIRS`` (extra folders to scan for already-scored keys, comma-separated),
``GP_FAITH_START_BATCH`` (1-indexed first batch; default **31** on Kaggle = rows **15,001-18,180**),
``GP_FAITH_LOG_CTX``, ``GP_FAITH_FORCE_BATCH`` (re-score one batch index),
``GP_PRIMARY_RAG_*`` (override primary-index tuning: ``RAG_TOP_K``, ``RAG_FETCH_MULT``, etc.),
``GP_FAITH_BEST_CONTEXT`` (default 1; PubMedQA gold never overridden),
``GP_FAITH_MEDQA_BLEND`` (default 1: blend re-retrieved + stored for MedQA/MMLU),
``GP_FAITH_CALIBRATE`` (default 0: mild score fix when judge clusters at 20),
``GP_FAITH_MERGE_MIN_SCORE`` / ``GP_FAITH_MERGE_MAX_GAP`` (merge gating),
``GP_FAITH_STORED_SLIGHT_MARGIN`` (default 1.12),
``GP_FAITH_OPTION_RETRIEVE``, ``GP_FAITH_LEXICAL_FALLBACK``, ``GP_FAITH_USE_HYBRID_RAG``,
``GP_FAITH_AUTO``, ``HF_TOKEN``. Call ``run_faithfulness_full_coverage()`` for 18,180 RAG rows.

**Paper run (reretrieve + strong judge):**::

    apply_paper_run_env()  # or apply_paper_run_env(rescore_all=True)
    run_faithfulness_paper_run()

Writes batches under ``Faithfulness/paper/`` (does not read old ``fathfullness/`` batches).

**DeepEval validation (no human audit):** stratified sample vs prior CSV + paper judge::

    run_deepeval_validation(sample_size=500)

Set ``GP_FAITH_VALIDATION_DEEPEVAL=1`` so Kaggle does not coerce DeepEval to local_hf.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

try:
    import requests
except ImportError:
    requests = None  # type: ignore


def _script_dir() -> str:
    """Notebook-safe base dir (__file__ is undefined when pasted into a Kaggle cell)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        if os.path.isdir("/kaggle/working"):
            return "/kaggle/working"
        return os.getcwd()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_BATCH_SIZE = 500
_DEFAULT_FAITHFULNESS_MODEL = "Qwen/Qwen3-8B"
_KAGGLE_DEFAULT_FAITHFULNESS_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_FALLBACK_FAITHFULNESS_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_KAGGLE_FAITH_WRITE_DIR = "/kaggle/working/Faithfulness"
_DEFAULT_FAITH_DIR = os.path.join(_script_dir(), "Faithfulness")
_KAGGLE_PREDICTIONS_CSV = (
    "/kaggle/input/datasets/fatinshadab/benchmark-results-all-predictions-combined/"
    "benchmark_results_all_predictions_combined.csv"
)
_KAGGLE_PREDICTIONS_ROW_OFFSET = 0
_KAGGLE_LAST_ROWS = 0
_KAGGLE_START_BATCH = 31  # 1-indexed: rows 15001-18180 (batches 31-37; batches 1-30 = rows 1-15000)
_KAGGLE_START_RAG_ROW = 15001
_KAGGLE_REMAINING_RAG_ROWS = 3180  # 18180 - 15000
_TARGET_RAG_ROWS = 18180
_TARGET_NORAG_ROWS = 18180
_TARGET_ALL_ROWS = 36360
_KAGGLE_RAG_DATASET_PRIMARY = "/kaggle/input/datasets/fatinshadab/rag-index"
_KAGGLE_RAG_DATASET_GOLD = "/kaggle/input/datasets/fatinshadab/rag-index-gold"
_KAGGLE_RAG_INDEX_SUBDIR = "rag_index"
_KAGGLE_RAG_GOLD_SUBDIR = "rag_index_gold"
_KAGGLE_RAG_FAISS_PRIMARY = os.path.join(
    _KAGGLE_RAG_DATASET_PRIMARY, _KAGGLE_RAG_INDEX_SUBDIR, "index.faiss"
)
_KAGGLE_RAG_FAISS_GOLD = os.path.join(
    _KAGGLE_RAG_DATASET_GOLD, _KAGGLE_RAG_GOLD_SUBDIR, "index.faiss"
)
# fatinshadab: resolved index dirs (chunks.jsonl + index.faiss under subfolders)
_KAGGLE_RAG_INDEX_FATINSHADAB = os.path.join(
    _KAGGLE_RAG_DATASET_PRIMARY, _KAGGLE_RAG_INDEX_SUBDIR
)
_KAGGLE_RAG_GOLD_INDEX_FATINSHADAB = os.path.join(
    _KAGGLE_RAG_DATASET_GOLD, _KAGGLE_RAG_GOLD_SUBDIR
)
_FATINSHADAB_PAPER_START_BATCH = 11  # resume: batches 1-10 done (rows 1-5000)
# ummesalmahabiba account (rag_index/ + rag_index_gold/ subfolders, same layout as fatinshadab)
_UMMESALMAHABIBA_RAG_DATASET_PRIMARY = "/kaggle/input/datasets/ummesalmahabiba/rag-index"
_UMMESALMAHABIBA_RAG_DATASET_GOLD = "/kaggle/input/datasets/ummesalmahabiba/rag-index-gold"
_KAGGLE_RAG_INDEX_UMMESALMAHABIBA = os.path.join(
    _UMMESALMAHABIBA_RAG_DATASET_PRIMARY, _KAGGLE_RAG_INDEX_SUBDIR
)
_KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA = os.path.join(
    _UMMESALMAHABIBA_RAG_DATASET_GOLD, _KAGGLE_RAG_GOLD_SUBDIR
)
_UMMESALMAHABIBA_PREDICTIONS_CSV = (
    "/kaggle/input/datasets/ummesalmahabiba/benchmark-results-all-predictions-combined/"
    "benchmark_results_all_predictions_combined.csv"
)
_UMMESALMAHABIBA_PAPER_START_BATCH = 36  # resume: batches 1-35 done (rows 1-17500)
_UMMESALMAHABIBA_GAP_FILL_PREDICTIONS = (
    "/kaggle/input/datasets/ummesalmahabiba/missing-rag-predictions/"
    "missing_rag_predictions.csv"
)
_UMMESALMAHABIBA_GAP_FILL_OUT_DIR = "/kaggle/working/fathfullness/rerun/gap_fill_umme"
_UMMESALMAHABIBA_GAP_FILL_START_BATCH = 6  # full 4785 CSV: batches 1–5 done elsewhere → 6/10
_UMMESALMAHABIBA_GAP_FILL_ROWS = 2500  # rows 2501–4785 ≈ last 5 batch slices of 4785 list
# sifatali008 account (flat rag-index/ layout: index.faiss + chunks.jsonl at dataset root)
_SIFATALI_PREDICTIONS_CSV = (
    "/kaggle/input/datasets/sifatali008/dataset/benchmark_results_all_predictions_combined.csv"
)
_SIFATALI_PAPER_START_BATCH = 11  # resume: batches 1-10 done elsewhere (rows 1-5000)
_KAGGLE_RAG_INDEX_SIFATALI = "/kaggle/input/datasets/sifatali008/rag-index"
_KAGGLE_RAG_GOLD_SIFATALI = "/kaggle/input/datasets/sifatali008/rag-index-gold"
# fahim220 account (flat rag-index/ at dataset root)
_FAHIM220_PREDICTIONS_CSV = (
    "/kaggle/input/datasets/fahim220/benchmark-results-all-predictions-combined/"
    "benchmark_results_all_predictions_combined.csv"
)
_FAHIM220_PAPER_START_BATCH = 16  # resume: batches 1-15 done (rows 1-7500)
_KAGGLE_RAG_INDEX_FAHIM220 = "/kaggle/input/datasets/fahim220/rag-index"
_KAGGLE_RAG_GOLD_FAHIM220 = "/kaggle/input/datasets/fahim220/rag-index-gold"
_FAHIM220_GAP_FILL_PREDICTIONS = (
    "/kaggle/input/datasets/fahim220/missing-rag-predictions-csv/"
    "missing_rag_predictions.csv"
)
_FAHIM220_GAP_FILL_OUT_DIR = "/kaggle/working/fathfullness/rerun/gap_fill"


def _local_rag_primary_dir() -> str:
    here = _script_dir()
    if os.path.basename(here) == "kaggle_working":
        return os.path.join(here, "rag_index")
    return os.path.join(here, "kaggle_working", "rag_index")


def _local_rag_gold_dir() -> str:
    here = _script_dir()
    if os.path.basename(here) == "kaggle_working":
        return os.path.join(here, "rag_index_gold")
    return os.path.join(here, "kaggle_working", "rag_index_gold")

_DEFAULT_PREDICTIONS_CANDIDATES = (
    _KAGGLE_PREDICTIONS_CSV,
    "/kaggle/input/datasets/fahim220/benchmark-results-all-predictions-combined/benchmark_results_all_predictions_combined.csv",
    "/kaggle/input/datasets/sifatali008/dataset/benchmark_results_all_predictions_combined.csv",
    "/kaggle/input/datasets/hafijur222/dataset/benchmark_results_all_predictions_combined.csv",
    "/kaggle/working/benchmark_results_all_predictions_combined.csv",
    os.path.join(_script_dir(), "result", "benchmark_results_all_predictions_combined.csv"),
)

_BUNDLED_SAMPLE_ROWS: List[Dict[str, Any]] = [
    {
        "id": "ex-high",
        "question": "Does metformin help type 2 diabetes?",
        "context": (
            "Randomized trials show metformin lowers HbA1c in adults with type 2 diabetes "
            "when used with diet and exercise."
        ),
        "answer": (
            "Yes. Metformin reduces HbA1c in type 2 diabetes alongside lifestyle measures, "
            "consistent with trial evidence."
        ),
    },
    {
        "id": "ex-low",
        "question": "Does metformin help type 2 diabetes?",
        "context": (
            "Randomized trials show metformin lowers HbA1c in adults with type 2 diabetes "
            "when used with diet and exercise."
        ),
        "answer": (
            "Metformin permanently cures all diabetes in every patient within one week "
            "with no side effects."
        ),
    },
]

_DEEPEVAL_METRICS: Dict[str, Any] = {}
_LOCAL_HF_MODELS: Dict[str, Tuple[Any, Any]] = {}


def _filtered_cli_args() -> List[str]:
    a = sys.argv[1:]
    out: List[str] = []
    i = 0
    while i < len(a):
        if a[i] == "-f" and i + 1 < len(a):
            i += 2
            continue
        out.append(a[i])
        i += 1
    return out


def _is_csv_path(path: str) -> bool:
    return path.lower().endswith(".csv")


def _default_batch_size() -> int:
    env = os.environ.get("GP_FAITH_BATCH_SIZE", "").strip()
    if env.isdigit():
        return int(env)
    if os.path.isdir("/kaggle"):
        return _DEFAULT_BATCH_SIZE
    return 0


def _faith_strict() -> bool:
    return os.environ.get("GP_FAITH_STRICT", "").strip().lower() in ("1", "true", "yes", "y")


def _kaggle_faithfulness_active() -> bool:
    return os.path.isdir("/kaggle") and not _faith_strict()


def _is_heavy_local_judge_model(model_id: str) -> bool:
    m = (model_id or "").strip().lower()
    if not m:
        return False
    heavy = (
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-32b",
        "qwen2.5-72b",
        "qwen2.5-32b",
        "llama-3.1-70b",
        "llama-3-70b",
    )
    return any(h in m for h in heavy)


def _normalize_faithfulness_run(backend: str, model_id: str = "") -> Tuple[str, str]:
    """
    On Kaggle T4, coerce deepeval + large HF judges to local_hf + Qwen2.5-3B.

    Set ``GP_FAITH_STRICT=1`` to keep ``GP_FAITHFULNESS_BACKEND`` / ``GP_FAITHFULNESS_MODEL`` as-is.
    Set ``GP_FAITH_VALIDATION_DEEPEVAL=1`` for ``run_deepeval_validation()`` (keeps DeepEval backend).
    """
    b = (backend or "").strip().lower() or "auto"
    m = (model_id or "").strip()
    if _env_truthy("GP_FAITH_VALIDATION_DEEPEVAL", default=False):
        return b, m
    if not _kaggle_faithfulness_active():
        return b, m

    new_b, new_m = b, m
    if b in ("deepeval", "auto"):
        new_b = "local_hf"
    if _is_heavy_local_judge_model(m):
        new_m = _KAGGLE_DEFAULT_FAITHFULNESS_MODEL

    if new_b != b or new_m != m:
        resolved = _resolve_faithfulness_model(new_m) if new_m else _resolve_faithfulness_model("")
        print(
            "[faithfulness] Kaggle T4: using "
            f"backend={new_b!r} model={resolved!r} "
            f"(requested backend={b!r} model={m or '(default)'}). "
            "Set GP_FAITH_STRICT=1 to disable coercion.",
            flush=True,
        )
    return new_b, new_m


def _default_backend() -> str:
    env = os.environ.get("GP_FAITHFULNESS_BACKEND", "").strip().lower()
    if env in ("deepeval", "openrouter", "local_hf", "auto"):
        if env != "auto":
            if _kaggle_faithfulness_active() and env == "deepeval":
                return "local_hf"
        return env
    if os.path.isdir("/kaggle"):
        return "local_hf"
    return "auto"


def _resolve_faithfulness_model(model: str = "") -> str:
    explicit = (model or "").strip()
    env = os.environ.get("GP_FAITHFULNESS_MODEL", "").strip()
    raw = explicit or env
    if _kaggle_faithfulness_active() and _is_heavy_local_judge_model(raw):
        return _KAGGLE_DEFAULT_FAITHFULNESS_MODEL
    if explicit:
        return explicit
    if env:
        return env
    if os.path.isdir("/kaggle"):
        return _KAGGLE_DEFAULT_FAITHFULNESS_MODEL
    return _DEFAULT_FAITHFULNESS_MODEL


def _default_max_ctx_chars() -> int:
    env = os.environ.get("GP_FAITH_MAX_CTX_CHARS", "").strip()
    if env.isdigit():
        return int(env)
    return 7500


def _env_truthy(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "n", "off"):
        return False
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    return default


def _default_mcq_only() -> bool:
    return _env_truthy("GP_FAITH_MCQ_ONLY", default=True)


def _default_strip_prompt() -> bool:
    return _env_truthy("GP_FAITH_STRIP_PROMPT", default=True)


def _default_faith_context_mode() -> str:
    """
    How to build CONTEXT for the judge.

    - ``stored``: use ``retrieved_context`` from predictions CSV (original behavior)
    - ``reretrieve``: always fetch fresh evidence from ``rag_index`` (MedQA / general)
    - ``gold``: PubMedQA → ``rag_index_gold``; other benchmarks → ``rag_index``
    - ``reretrieve_or_stored`` (default when indexes exist): fresh retrieval if non-empty,
      else fall back to stored CSV context
    """
    env = os.environ.get("GP_FAITH_CONTEXT", "").strip().lower()
    if env in ("stored", "reretrieve", "gold", "reretrieve_or_stored"):
        return env
    if _resolve_faith_rag_primary_dir() or _resolve_faith_rag_gold_dir():
        return "reretrieve_or_stored"
    return "stored"


def _normalize_rag_index_dir(path: str) -> str:
    """Accept index folder or a direct path to ``index.faiss``."""
    p = (path or "").strip()
    if not p:
        return ""
    p = os.path.abspath(p)
    base = os.path.basename(p).lower()
    if base == "index.faiss" and os.path.isfile(p):
        return os.path.dirname(p)
    if base == "chunks.jsonl" and os.path.isfile(p):
        return os.path.dirname(p)
    return p


def _rag_index_ready(index_dir: str) -> bool:
    d = _normalize_rag_index_dir(index_dir)
    if not d:
        return False
    cp = os.path.join(d, "chunks.jsonl")
    ip = os.path.join(d, "index.faiss")
    return os.path.isfile(cp) and (os.path.isfile(ip) or os.path.getsize(cp) > 0)


def _discover_rag_index_dir(path: str, *, gold: bool = False) -> str:
    """
    Resolve a Kaggle dataset root or folder to ``rag_index/`` or ``rag_index_gold/``.

    Accepts dataset roots (``.../fatinshadab/rag-index``), subfolders, or direct
    paths to ``index.faiss`` / ``chunks.jsonl``.
    """
    root = _normalize_rag_index_dir(path)
    if not root:
        return ""
    sub = _KAGGLE_RAG_GOLD_SUBDIR if gold else _KAGGLE_RAG_INDEX_SUBDIR
    candidates = [root]
    if os.path.basename(root.rstrip(os.sep)) not in (sub,):
        candidates.append(os.path.join(root, sub))
    for cand in candidates:
        d = _normalize_rag_index_dir(cand)
        if _rag_index_ready(d):
            return os.path.abspath(d)
    return ""


def _apply_default_kaggle_rag_env() -> None:
    """On Kaggle, auto-pin RAG dirs from attached fatinshadab or sifatali008 datasets."""
    if not os.path.isdir("/kaggle/input"):
        return
    account = os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower()
    if account == "sifatali008":
        primary_roots = (
            _KAGGLE_RAG_INDEX_SIFATALI,
            "/kaggle/input/sifatali008/rag-index",
            "/kaggle/input/datasets/sifatali008/rag-index",
        )
        gold_roots = (
            _KAGGLE_RAG_GOLD_SIFATALI,
            "/kaggle/input/sifatali008/rag-index-gold",
            "/kaggle/input/datasets/sifatali008/rag-index-gold",
        )
    elif account == "fahim220":
        primary_roots = (
            _KAGGLE_RAG_INDEX_FAHIM220,
            "/kaggle/input/fahim220/rag-index",
            "/kaggle/input/datasets/fahim220/rag-index",
        )
        gold_roots = (
            _KAGGLE_RAG_GOLD_FAHIM220,
            "/kaggle/input/fahim220/rag-index-gold",
            "/kaggle/input/datasets/fahim220/rag-index-gold",
        )
    elif account in ("fatinshadab", ""):
        primary_roots = (
            _KAGGLE_RAG_INDEX_FATINSHADAB,
            _KAGGLE_RAG_FAISS_PRIMARY,
            _KAGGLE_RAG_DATASET_PRIMARY,
            "/kaggle/input/datasets/fatinshadab/rag-index",
            "/kaggle/input/fatinshadab/rag-index",
        )
        gold_roots = (
            _KAGGLE_RAG_GOLD_INDEX_FATINSHADAB,
            _KAGGLE_RAG_FAISS_GOLD,
            _KAGGLE_RAG_DATASET_GOLD,
            "/kaggle/input/datasets/fatinshadab/rag-index-gold",
            "/kaggle/input/fatinshadab/rag-index-gold",
        )
    elif account == "ummesalmahabiba":
        primary_roots = (
            _KAGGLE_RAG_INDEX_UMMESALMAHABIBA,
            _UMMESALMAHABIBA_RAG_DATASET_PRIMARY,
            "/kaggle/input/datasets/ummesalmahabiba/rag-index/rag_index",
            "/kaggle/input/datasets/ummesalmahabiba/rag-index",
            "/kaggle/input/ummesalmahabiba/rag-index",
        )
        gold_roots = (
            _KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA,
            _UMMESALMAHABIBA_RAG_DATASET_GOLD,
            "/kaggle/input/datasets/ummesalmahabiba/rag-index-gold/rag_index_gold",
            "/kaggle/input/datasets/ummesalmahabiba/rag-index-gold",
            "/kaggle/input/ummesalmahabiba/rag-index-gold",
        )
    else:
        primary_roots = (
            _KAGGLE_RAG_INDEX_FATINSHADAB,
            _KAGGLE_RAG_DATASET_PRIMARY,
            _KAGGLE_RAG_FAISS_PRIMARY,
            _KAGGLE_RAG_INDEX_SIFATALI,
            "/kaggle/input/fatinshadab/rag-index",
            "/kaggle/input/sifatali008/rag-index",
        )
        gold_roots = (
            _KAGGLE_RAG_GOLD_INDEX_FATINSHADAB,
            _KAGGLE_RAG_DATASET_GOLD,
            _KAGGLE_RAG_FAISS_GOLD,
            _KAGGLE_RAG_GOLD_SIFATALI,
            "/kaggle/input/fatinshadab/rag-index-gold",
            "/kaggle/input/sifatali008/rag-index-gold",
        )
    for env_key, roots, is_gold in (
        ("GP_FAITH_RAG_INDEX_DIR", primary_roots, False),
        ("GP_FAITH_RAG_GOLD_DIR", gold_roots, True),
    ):
        if os.environ.get(env_key, "").strip():
            continue
        for root in roots:
            d = _discover_rag_index_dir(root, gold=is_gold)
            if not d and isinstance(root, str) and root.endswith(".faiss") and os.path.isfile(root):
                d = _discover_rag_index_dir(root, gold=is_gold)
            if d:
                os.environ[env_key] = d
                break


def _apply_default_sifatali_rag_env() -> None:
    """Backward-compatible alias."""
    _apply_default_kaggle_rag_env()


_RAG_ENV_KEYS = (
    "RAG_INDEX_DIR",
    "RAG_FAISS_INDEX",
    "RAG_CHUNKS_JSONL",
    "RAG_TOP_K",
    "RAG_CONTEXT_MAX_CHARS",
    "RAG_EMBED_MODEL",
    "RAG_AUTO_BUILD",
    "RAG_FORCE_MOCK",
    "RAG_MIN_DENSE_SCORE",
    "RAG_FETCH_MULT",
    "RAG_RERANK_CANDIDATES",
)

_RMR_MODULE: Any = None


def _rag_env_snapshot() -> Dict[str, str]:
    return {k: os.environ.get(k, "") for k in _RAG_ENV_KEYS}


def _pin_rag_environment(index_dir: str) -> str:
    d = os.path.abspath(index_dir.strip())
    os.environ["RAG_INDEX_DIR"] = d
    os.environ["RAG_FAISS_INDEX"] = os.path.join(d, "index.faiss")
    os.environ["RAG_CHUNKS_JSONL"] = os.path.join(d, "chunks.jsonl")
    if not os.environ.get("RAG_EMBED_MODEL", "").strip():
        os.environ["RAG_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["RAG_AUTO_BUILD"] = "0"
    if os.environ.get("GP_FAITH_USE_HYBRID_RAG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        os.environ.pop("RAG_RETRIEVAL_LEGACY", None)
        os.environ.setdefault("RAG_DISABLE_RRF", "0")
    else:
        os.environ.setdefault("RAG_RETRIEVAL_LEGACY", "1")
    return d


@contextlib.contextmanager
def _temporary_rag_corpus(index_dir: str):
    saved = _rag_env_snapshot()
    try:
        _pin_rag_environment(index_dir)
        _clear_rag_cache()
        yield
    finally:
        for k, prev in saved.items():
            if prev:
                os.environ[k] = prev
            else:
                os.environ.pop(k, None)
        _clear_rag_cache()


@contextlib.contextmanager
def _pubmedqa_gold_retrieval_tuning():
    saved: Dict[str, str] = {}
    for key, default in (
        ("RAG_MIN_DENSE_SCORE", "0.12"),
        ("RAG_FETCH_MULT", "8"),
        ("RAG_RERANK_CANDIDATES", "30"),
    ):
        new_val = os.environ.get(f"GP_GOLD_{key}", default).strip() or default
        saved[key] = os.environ.get(key, "")
        os.environ[key] = new_val
    try:
        yield
    finally:
        for key, prev in saved.items():
            if prev:
                os.environ[key] = prev
            else:
                os.environ.pop(key, None)


_FAITH_PRIMARY_TUNING_LOGGED = False


@contextlib.contextmanager
def _faithfulness_primary_retrieval_tuning():
    """Tighter retrieval for MedQA/MMLU on primary index (faithfulness grounding)."""
    global _FAITH_PRIMARY_TUNING_LOGGED
    saved: Dict[str, str] = {}
    for key, default in (
        ("RAG_TOP_K", "6"),
        ("RAG_CONTEXT_MAX_CHARS", "7500"),
        ("RAG_FETCH_MULT", "12"),
        ("RAG_LEXICAL_WEIGHT", "0.40"),
        ("RAG_MIN_DENSE_SCORE", "0.08"),
        ("RAG_DEDUP_JACCARD", "0.85"),
    ):
        new_val = os.environ.get(f"GP_PRIMARY_{key}", default).strip() or default
        saved[key] = os.environ.get(key, "")
        os.environ[key] = new_val
    if not _FAITH_PRIMARY_TUNING_LOGGED:
        _FAITH_PRIMARY_TUNING_LOGGED = True
        print(
            "[faithfulness] Primary-index retrieval tuning: "
            f"top_k={os.environ['RAG_TOP_K']}, fetch_mult={os.environ['RAG_FETCH_MULT']}, "
            f"lex_weight={os.environ['RAG_LEXICAL_WEIGHT']}, "
            f"min_dense={os.environ['RAG_MIN_DENSE_SCORE']}",
            flush=True,
        )
    try:
        yield
    finally:
        for key, prev in saved.items():
            if prev:
                os.environ[key] = prev
            else:
                os.environ.pop(key, None)


@contextlib.contextmanager
def _quiet_ml_load():
    """Suppress noisy HF/ST load reports (e.g. UNEXPECTED position_ids) on Kaggle."""
    import logging
    import warnings

    names = (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "torch",
    )
    prev = {n: logging.getLogger(n).level for n in names}
    try:
        for n in names:
            logging.getLogger(n).setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for n, lvl in prev.items():
            logging.getLogger(n).setLevel(lvl)


def _clear_rag_cache() -> None:
    global _INLINE_RAG_CACHE
    _INLINE_RAG_CACHE.clear()
    _INLINE_INDEX_BY_DIR.clear()
    try:
        import real_model_runner as rmr

        if hasattr(rmr, "_clear_rag_cache"):
            rmr._clear_rag_cache()
    except Exception:
        pass


_INLINE_RAG_CACHE: Dict[str, Any] = {}
_INLINE_EMBED_CACHE: Dict[str, Any] = {}
_INLINE_INDEX_BY_DIR: Dict[str, Dict[str, Any]] = {}
_INLINE_INDEX_LOAD_LOGGED: set = set()
_INLINE_EMBED_LOGGED: set = set()
_INLINE_RAG_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)
_INLINE_RAG_BACKEND_LOGGED = False
_FAITH_CONTEXT_SETUP_LOGGED = False


def _inline_rag_deps_ok() -> bool:
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _inline_rag_retrieval_available() -> bool:
    if not _inline_rag_deps_ok():
        return False
    return bool(_resolve_faith_rag_primary_dir() or _resolve_faith_rag_gold_dir())


def _inline_load_chunk_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"text": line, "source": ""})
                continue
            if isinstance(obj, dict):
                t = obj.get("text") or obj.get("chunk") or obj.get("content") or ""
                rows.append(
                    {
                        "text": str(t),
                        "source": str(obj.get("source") or obj.get("id") or ""),
                    }
                )
            else:
                rows.append({"text": str(obj), "source": ""})
    return rows


def _inline_rag_token_set(text: str) -> set:
    return set(_INLINE_RAG_TOKEN_RE.findall((text or "").lower()))


def _inline_rag_token_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _inline_lexical_score(query: str, body: str, q_toks: Optional[set] = None) -> float:
    q_toks = q_toks or _inline_rag_token_set(query)
    if not q_toks:
        return 0.0
    body = (body or "").strip()
    if not body:
        return 0.0
    ql = (query or "").lower()
    toks = _inline_rag_token_set(body)
    jac = _inline_rag_token_jaccard(q_toks, toks)
    overlap = len(q_toks & toks)
    phrase_bonus = 0.0
    for t in sorted(q_toks, key=len, reverse=True):
        if len(t) >= 5 and t in ql and t in body.lower():
            phrase_bonus += 0.02
    return jac + 0.08 * min(overlap, 40) + phrase_bonus


def _inline_format_evidence(hits: List[Dict[str, Any]], max_chars: int) -> str:
    lines: List[str] = []
    used = 0
    for rank, hit in enumerate(hits, start=1):
        body = str(hit.get("body") or "")
        meta = [f"idx={int(hit['idx'])}"]
        if hit.get("dense_score") is not None:
            meta.append(f"dense={float(hit['dense_score']):.4f}")
        if hit.get("lexical_score") is not None:
            meta.append(f"lexical={float(hit['lexical_score']):.4f}")
        src = str(hit.get("source") or "")
        if src:
            meta.append(f"source={src}")
        chunk_line = f"Chunk {rank} ({', '.join(meta)}):\n{body}"
        if used + len(chunk_line) + 2 > max_chars:
            remain = max_chars - used - 80
            if remain <= 80:
                break
            chunk_line = f"Chunk {rank} ({', '.join(meta)}):\n{body[:remain]}..."
        lines.append(chunk_line)
        used += len(chunk_line) + 2
    if not lines:
        return ""
    header = f"Retrieved Medical Evidence:\n\nRETRIEVED EVIDENCE (FAISS top-{len(lines)}):\n\n"
    return header + "\n\n".join(lines)


def _inline_get_embedder(embed_name: str, device: str) -> Any:
    """Load sentence-transformers embedder once per (model, device); reused across indexes."""
    key = f"{embed_name}|{device}"
    if key in _INLINE_EMBED_CACHE:
        return _INLINE_EMBED_CACHE[key]
    from sentence_transformers import SentenceTransformer

    with _quiet_ml_load():
        try:
            embedder = SentenceTransformer(embed_name, device=device)
        except TypeError:
            embedder = SentenceTransformer(embed_name)
    _INLINE_EMBED_CACHE[key] = embedder
    if key not in _INLINE_EMBED_LOGGED:
        _INLINE_EMBED_LOGGED.add(key)
        print(
            f"[faithfulness] RAG embedder ready: {embed_name!r} (device={device})",
            flush=True,
        )
    return embedder


def _inline_activate_index_cache(index_dir: str) -> bool:
    """Point active retrieval state at a cached index directory."""
    global _INLINE_RAG_CACHE
    d = _normalize_rag_index_dir(index_dir)
    entry = _INLINE_INDEX_BY_DIR.get(d)
    if not entry:
        return False
    _INLINE_RAG_CACHE = {"index_dir": d, **entry}
    return True


def _inline_load_rag_index(index_dir: str) -> bool:
    global _INLINE_RAG_CACHE
    d = _normalize_rag_index_dir(index_dir)
    if not d or not _rag_index_ready(d):
        return False
    if d in _INLINE_INDEX_BY_DIR:
        _inline_activate_index_cache(d)
        return True
    idx_path = os.path.join(d, "index.faiss")
    chunk_path = os.path.join(d, "chunks.jsonl")
    embed_name = (
        os.environ.get("RAG_EMBED_MODEL", "").strip()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    device = (os.environ.get("RAG_EMBED_DEVICE") or "cpu").strip() or "cpu"
    try:
        import faiss
    except ImportError:
        return False
    try:
        embedder = _inline_get_embedder(embed_name, device)
        with _quiet_ml_load():
            index = faiss.read_index(idx_path)
        records = _inline_load_chunk_records(chunk_path)
        texts = [str(r.get("text") or "") for r in records]
        sources = [str(r.get("source") or "") for r in records]
        if index.ntotal != len(texts):
            if len(texts) > index.ntotal:
                texts = texts[: index.ntotal]
                sources = sources[: index.ntotal]
            else:
                pad = index.ntotal - len(texts)
                texts.extend([""] * pad)
                sources.extend([""] * pad)
        entry = {
            "index": index,
            "texts": texts,
            "sources": sources,
            "st": embedder,
        }
        _INLINE_INDEX_BY_DIR[d] = entry
        _INLINE_RAG_CACHE = {"index_dir": d, **entry}
        if d not in _INLINE_INDEX_LOAD_LOGGED:
            _INLINE_INDEX_LOAD_LOGGED.add(d)
            print(
                f"[faithfulness] loaded FAISS index: {d} "
                f"(vectors={index.ntotal}, chunks={len(texts)})",
                flush=True,
            )
        return True
    except Exception as exc:
        print(f"[faithfulness] inline RAG load failed: {exc}", flush=True)
        return False


def _inline_retrieve_faiss(query: str) -> str:
    """FAISS dense + lexical blend using env RAG_INDEX_DIR (paste-only notebook path)."""
    index_dir = os.environ.get("RAG_INDEX_DIR", "").strip()
    if not _inline_load_rag_index(index_dir):
        return ""
    import faiss
    import numpy as np

    top_k = int(os.environ.get("RAG_TOP_K", "3") or "3")
    max_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "4500") or "4500")
    w_lex = float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.22") or "0.22")
    w_lex = max(0.0, min(1.0, w_lex))
    fetch_mult = max(1, int(os.environ.get("RAG_FETCH_MULT", "6") or "6"))
    min_dense = float(os.environ.get("RAG_MIN_DENSE_SCORE", "0") or "0")
    dedup_thr = float(os.environ.get("RAG_DEDUP_JACCARD", "0.88") or "0.88")
    dedup_thr = max(0.0, min(1.0, dedup_thr))

    index = _INLINE_RAG_CACHE["index"]
    texts: List[str] = _INLINE_RAG_CACHE["texts"]
    sources: List[str] = _INLINE_RAG_CACHE.get("sources") or [""] * len(texts)
    embedder = _INLINE_RAG_CACHE["st"]

    enc_kw: Dict[str, Any] = {"convert_to_numpy": True, "show_progress_bar": False}
    try:
        try:
            qv = embedder.encode(
                [query],
                normalize_embeddings=True,
                **enc_kw,
            )
        except TypeError:
            qv = embedder.encode([query], **enc_kw)
        if qv.dtype != np.float32:
            qv = qv.astype(np.float32)
        faiss.normalize_L2(qv)
        ntotal = int(index.ntotal)
        fetch_k = min(max(top_k * fetch_mult, top_k + 4), ntotal, max(1, ntotal))
        sims, ids = index.search(qv, fetch_k)
        q_toks = _inline_rag_token_set(query)
        scored: List[Tuple[float, float, float, int, str, str]] = []
        for rank, j in enumerate(ids[0]):
            if j < 0 or j >= len(texts):
                continue
            jj = int(j)
            body = (texts[jj] or "").strip()
            if not body:
                continue
            dense = float(sims[0][rank]) if rank < len(sims[0]) else 0.0
            if dense != dense:
                dense = 0.0
            lex = _inline_lexical_score(query, body, q_toks)
            if min_dense > 0 and dense < min_dense and lex < 0.08:
                continue
            combined = (1.0 - w_lex) * dense + w_lex * lex
            body_toks = _inline_rag_token_set(body)
            scored.append(
                (combined, dense, lex, jj, body, sources[jj] if jj < len(sources) else "", body_toks)
            )
        scored.sort(key=lambda t: -t[0])
        hits: List[Dict[str, Any]] = []
        kept_tok_sets: List[set] = []
        for comb, dense, lex, jj, body, src, body_toks in scored:
            if len(hits) >= top_k:
                break
            if kept_tok_sets and any(
                _inline_rag_token_jaccard(body_toks, kt) >= dedup_thr for kt in kept_tok_sets
            ):
                continue
            hits.append(
                {
                    "idx": jj,
                    "body": body,
                    "source": src,
                    "dense_score": dense,
                    "lexical_score": lex,
                    "combined_score": comb,
                }
            )
            kept_tok_sets.append(body_toks)
        block = _inline_format_evidence(hits, max_chars)
        max_comb = max((h.get("combined_score") or 0) for h in hits) if hits else 0.0
        if max_comb < 0.22 and os.environ.get("GP_FAITH_LEXICAL_FALLBACK", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            lex_hits = _inline_lexical_fallback_hits(query, top_k=top_k, texts=texts, sources=sources)
            if lex_hits:
                block = _merge_evidence_text(block, _inline_format_evidence(lex_hits, max_chars // 2))
        return block[:max_chars] if block else ""
    except Exception as exc:
        print(f"[faithfulness] inline FAISS search failed: {exc}", flush=True)
        return ""


def _inline_lexical_fallback_hits(
    query: str,
    *,
    top_k: int,
    texts: List[str],
    sources: List[str],
) -> List[Dict[str, Any]]:
    """Lexical top-k over full corpus when dense retrieval is weak."""
    q_toks = _inline_rag_token_set(query)
    if not q_toks:
        return []
    scored: List[Tuple[float, int, str, str]] = []
    for i, raw in enumerate(texts):
        body = (raw or "").strip()
        if not body:
            continue
        lex = _inline_lexical_score(query, body, q_toks)
        if lex < 0.06:
            continue
        src = sources[i] if i < len(sources) else ""
        scored.append((lex, i, body, src))
    scored.sort(key=lambda t: -t[0])
    return [
        {
            "idx": i,
            "body": body,
            "source": src,
            "lexical_score": lex,
            "dense_score": 0.0,
            "combined_score": lex,
        }
        for lex, i, body, src in scored[:top_k]
    ]


def _merge_evidence_text(a: str, b: str) -> str:
    """Concatenate two evidence blocks, skipping duplicate chunk bodies."""
    if not a:
        return b
    if not b:
        return a
    seen: set = set()
    parts: List[str] = []
    for block in (a, b):
        for chunk in re.split(r"(?=Chunk \d+ \()", block):
            chunk = chunk.strip()
            if not chunk:
                continue
            key = chunk[:200].lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(chunk)
    header = "Retrieved Medical Evidence:\n\nRETRIEVED EVIDENCE (merged):\n\n"
    return header + "\n\n".join(parts)


def _context_support_score(context: str, answer: str, question: str = "") -> float:
    """
    Score how well CONTEXT supports judging the answer (not boilerplate overlap).
    Weights question + answer; bonus for real evidence blocks with chunk citations.
    """
    ctx = (context or "").strip()
    if not ctx:
        return 0.0
    ans = (answer or "").strip()
    q = (question or "").strip()
    a_score = _inline_lexical_score(ans, ctx) if ans else 0.0
    q_score = _inline_lexical_score(q, ctx) if q else 0.0
    score = 0.55 * a_score + 0.45 * q_score
    if re.search(r"\[\d+\]", ctx):
        score += 0.08
    low = ctx.lower()
    if "retrieved medical evidence" in low or "retrieved evidence" in low:
        score += 0.04
    # Penalize query-only stubs with no chunk structure
    if re.search(r"Chunk \d+ \(", ctx):
        score += 0.05
    if len(ctx) < 200 and not re.search(r"\[\d+\]", ctx) and "Chunk" not in ctx:
        score *= 0.5
    return score


def _blend_reretrieve_stored_context(
    fresh: str,
    stored: str,
    *,
    max_chars: int,
    fresh_frac: float = 0.45,
) -> str:
    """Combine re-retrieved chunks with stored CSV evidence (MedQA/MMLU default path)."""
    fresh_frac = max(0.25, min(0.55, fresh_frac))
    f_cap = int(max_chars * fresh_frac)
    s_cap = max_chars - f_cap - 80
    return _merge_evidence_text(fresh[:f_cap], stored[:s_cap])[:max_chars]


def _calibrate_mcq_faithfulness_score(
    score: float,
    note: str,
    context: str,
    answer: str,
    question: str,
) -> Tuple[float, str]:
    """
    Mild correction when the judge clusters at 20 despite partial support in the note/context.
    Opt-in via GP_FAITH_CALIBRATE=1 (off by default for strict paper runs).
    """
    if os.environ.get("GP_FAITH_CALIBRATE", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return score, note
    if score != score or score > 25:
        return score, note
    support = _context_support_score(context, answer, question)
    n = (note or "").lower()
    partial_words = (
        "partial",
        "related",
        "same disease",
        "same drug",
        "mentions",
        "discusses",
        "consistent",
        "supports",
        "relevant",
        "mechanism",
        "class",
    )
    if support >= 0.08 and any(w in n for w in partial_words):
        return max(score, 38.0), note
    if support >= 0.12 and score <= 20:
        return max(score, 42.0), note
    return score, note


def _option_focused_retrieval_query(row: Dict[str, Any]) -> str:
    """Query from selected MCQ option text only (second retrieval pass)."""
    ans = _build_faithfulness_answer(row, mcq_only=True)
    m = re.search(r"Option text:\s*(.+)", ans, re.IGNORECASE | re.DOTALL)
    if m:
        return str(m.group(1)).strip()[:2000]
    return ans.strip()[:2000]


def _log_inline_rag_backend_once() -> None:
    global _INLINE_RAG_BACKEND_LOGGED
    if _INLINE_RAG_BACKEND_LOGGED:
        return
    _INLINE_RAG_BACKEND_LOGGED = True
    print(
        "[faithfulness] RAG re-retrieval: built-in FAISS (no real_model_runner.py). "
        "Requires faiss-cpu + sentence-transformers.",
        flush=True,
    )


def _find_real_model_runner_root() -> str:
    """Locate directory containing real_model_runner.py (no eval_benchmarks import)."""
    seen: set[str] = set()
    candidates: List[str] = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    for r in (
        _script_dir(),
        "/kaggle/working",
        os.path.join(_script_dir(), ".."),
        os.path.join(_script_dir(), "..", "GreenPaper_Kaggle_Benchmarks"),
    ):
        if r:
            ar = os.path.abspath(r)
            if ar not in seen:
                seen.add(ar)
                candidates.append(ar)
    for c in candidates:
        if os.path.isfile(os.path.join(c, "real_model_runner.py")):
            return c
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            for sub in ("", "dataset", os.path.join("datasets", name)):
                d = os.path.join(kin, name, sub) if sub else os.path.join(kin, name)
                if os.path.isfile(os.path.join(d, "real_model_runner.py")):
                    return os.path.abspath(d)
            root = os.path.join(kin, name)
            try:
                for dirpath, _dn, filenames in os.walk(root):
                    depth = dirpath[len(root) :].count(os.sep)
                    if depth > 5:
                        continue
                    if "real_model_runner.py" in filenames:
                        return os.path.abspath(dirpath)
            except OSError:
                pass
    return ""


_FAITH_RAG_HELPER_FILES = (
    "real_model_runner.py",
    "measurement_config.py",
    "eval_quality_metrics.py",
)


def _sync_faithfulness_rag_helpers(work_dir: str) -> bool:
    """Copy RAG helper modules into work_dir when missing (paste-only Kaggle notebooks)."""
    import shutil

    work = os.path.abspath(work_dir)
    os.makedirs(work, exist_ok=True)
    if os.path.isfile(os.path.join(work, "real_model_runner.py")):
        return True
    src_roots: List[str] = []
    for r in (
        _script_dir(),
        os.path.join(_script_dir(), "..", "GreenPaper_Kaggle_Benchmarks"),
        os.path.join(_script_dir(), "GreenPaper_Kaggle_Benchmarks"),
        "/kaggle/working",
    ):
        if r:
            ar = os.path.abspath(r)
            if ar not in src_roots:
                src_roots.append(ar)
    rmr_root = _find_real_model_runner_root()
    if rmr_root and rmr_root not in src_roots:
        src_roots.insert(0, rmr_root)
    copied = False
    for root in src_roots:
        for name in _FAITH_RAG_HELPER_FILES:
            src = os.path.join(root, name)
            dst = os.path.join(work, name)
            if os.path.isfile(src) and not os.path.isfile(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f"[faithfulness] synced {name} → {dst}", flush=True)
                    copied = True
                except OSError:
                    pass
        if os.path.isfile(os.path.join(work, "real_model_runner.py")):
            return True
    return copied or os.path.isfile(os.path.join(work, "real_model_runner.py"))


def _try_unpack_helpers_via_eval_benchmarks(work_dir: str) -> bool:
    """If eval_benchmarks.py exists in working, run its Kaggle helper sync."""
    for eb in (
        os.path.join(work_dir, "eval_benchmarks.py"),
        os.path.join(_script_dir(), "eval_benchmarks.py"),
    ):
        if not os.path.isfile(eb):
            continue
        try:
            root = os.path.dirname(os.path.abspath(eb))
            if root not in sys.path:
                sys.path.insert(0, root)
            import eval_benchmarks as eb_mod  # noqa: F401

            if hasattr(eb_mod, "_sync_kaggle_helper_bundle"):
                eb_mod._sync_kaggle_helper_bundle(work_dir)
            return os.path.isfile(os.path.join(work_dir, "real_model_runner.py"))
        except Exception:
            continue
    return False


def _try_import_real_model_runner() -> Any:
    global _RMR_MODULE
    if _RMR_MODULE is not None:
        return _RMR_MODULE
    if not _find_real_model_runner_root():
        return None
    try:
        return _import_real_model_runner()
    except ImportError:
        return None


def _import_real_model_runner():
    global _RMR_MODULE
    if _RMR_MODULE is not None:
        return _RMR_MODULE
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else _script_dir()
    if not os.path.isfile(os.path.join(work, "real_model_runner.py")):
        _sync_faithfulness_rag_helpers(work)
        _try_unpack_helpers_via_eval_benchmarks(work)
    root = _find_real_model_runner_root()
    if not root:
        raise ImportError(
            "real_model_runner.py not found for RAG re-retrieval. "
            "On Kaggle: copy GreenPaper_Kaggle_Benchmarks/real_model_runner.py to "
            "/kaggle/working/ (see Cell 0c in run_faithfulness_eval.py docstring), "
            "or save eval_benchmarks.py to /kaggle/working and run once."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    import real_model_runner as rmr

    _RMR_MODULE = rmr
    return rmr


def _prefetch_rag_for_query(query: str) -> str:
    rmr = _try_import_real_model_runner()
    if rmr is not None:
        try:
            block, _ev, _src = rmr.build_rag_context(query, True)
            if block:
                return str(block)
            ev = str(getattr(rmr, "LAST_RAG_EVIDENCE", "") or "")
            if ev:
                return ev
        except Exception as exc:
            print(f"[faithfulness] real_model_runner RAG note: {exc}", flush=True)
    if _inline_rag_deps_ok():
        _log_inline_rag_backend_once()
        block = _inline_retrieve_faiss(query)
        if block:
            return block
    return ""


def _prewarm_faith_rag(index_dir: str) -> None:
    if not index_dir:
        return
    try:
        with _temporary_rag_corpus(index_dir):
            rmr = _try_import_real_model_runner()
            if rmr is not None:
                if hasattr(rmr, "_try_autobuild_rag_index_once"):
                    rmr._try_autobuild_rag_index_once()
                if hasattr(rmr, "_retrieve_faiss"):
                    rmr._retrieve_faiss("warmup query", top_k=1, max_context_chars=64)
                    return
            if _inline_load_rag_index(index_dir):
                _inline_retrieve_faiss("warmup biomedical query")
    except Exception as exc:
        print(f"[faithfulness] RAG prewarm note: {exc}", flush=True)


def _rag_dir_from_env(env_key: str) -> str:
    """Use pinned env path when index files are present (even if discovery order fails)."""
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return ""
    d = _discover_rag_index_dir(raw, gold=(env_key == "GP_FAITH_RAG_GOLD_DIR"))
    if d:
        return d
    norm = _normalize_rag_index_dir(raw)
    if norm and _rag_index_ready(norm):
        return norm
    return ""


def _resolve_faith_rag_primary_dir() -> str:
    pinned = _rag_dir_from_env("GP_FAITH_RAG_INDEX_DIR")
    if pinned:
        return pinned
    for cand in (
        os.environ.get("GP_FAITH_RAG_INDEX_DIR", "").strip(),
        os.environ.get("RAG_INDEX_DIR", "").strip(),
        os.environ.get("RAG_FAISS_INDEX", "").strip(),
        _KAGGLE_RAG_INDEX_UMMESALMAHABIBA,
        _UMMESALMAHABIBA_RAG_DATASET_PRIMARY,
        _KAGGLE_RAG_INDEX_FATINSHADAB,
        _KAGGLE_RAG_FAISS_PRIMARY,
        _KAGGLE_RAG_DATASET_PRIMARY,
        "/kaggle/input/fatinshadab/rag-index",
        "/kaggle/input/datasets/fatinshadab/rag-index",
        _KAGGLE_RAG_INDEX_SIFATALI,
        "/kaggle/input/sifatali008/rag-index",
        _local_rag_primary_dir(),
        "/kaggle/working/rag_index",
        "/kaggle/input/datasets/salmashopna/rag-index",
        "/kaggle/input/salmashopna/rag-index",
        "/kaggle/input/datasets/hafijur222/rag-index",
    ):
        d = _discover_rag_index_dir(cand, gold=False)
        if d:
            return d
    return ""


def _resolve_faith_rag_gold_dir() -> str:
    pinned = _rag_dir_from_env("GP_FAITH_RAG_GOLD_DIR")
    if pinned:
        return pinned
    for cand in (
        os.environ.get("GP_FAITH_RAG_GOLD_DIR", "").strip(),
        os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip(),
        os.environ.get("GP_FAITH_RAG_GOLD_FAISS", "").strip(),
        _KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA,
        _UMMESALMAHABIBA_RAG_DATASET_GOLD,
        _KAGGLE_RAG_GOLD_INDEX_FATINSHADAB,
        _KAGGLE_RAG_FAISS_GOLD,
        _KAGGLE_RAG_DATASET_GOLD,
        "/kaggle/input/fatinshadab/rag-index-gold",
        "/kaggle/input/datasets/fatinshadab/rag-index-gold",
        _KAGGLE_RAG_GOLD_SIFATALI,
        "/kaggle/input/sifatali008/rag-index-gold",
        _local_rag_gold_dir(),
        "/kaggle/working/rag_index_gold",
        "/kaggle/input/datasets/salmashopna/rag-index-gold",
        "/kaggle/input/salmashopna/rag-index-gold",
    ):
        d = _discover_rag_index_dir(cand, gold=True)
        if d:
            return d
    return ""


def _log_pinned_rag_paths(primary: str, gold: str) -> None:
    for label, d in (("primary", primary), ("gold", gold)):
        if not d:
            continue
        print(
            f"[faithfulness] RAG {label}: dir={d!r} "
            f"faiss={os.path.join(d, 'index.faiss')!r} "
            f"exists={os.path.isfile(os.path.join(d, 'index.faiss'))}",
            flush=True,
        )


def _describe_rag_index_dir(index_dir: str) -> str:
    if not index_dir:
        return "missing"
    cp = os.path.join(index_dir, "chunks.jsonl")
    ip = os.path.join(index_dir, "index.faiss")
    parts = [f"chunks={os.path.isfile(cp)}", f"faiss={os.path.isfile(ip)}"]
    mf = os.path.join(index_dir, "pubmedqa_gold_manifest.json")
    if os.path.isfile(mf):
        parts.append("gold_manifest=yes")
    return ", ".join(parts)


def _warn_faithfulness_context_setup(context_mode: str) -> None:
    """Tell the user when scores will stay low (stored CSV / missing indexes)."""
    global _FAITH_CONTEXT_SETUP_LOGGED
    if _FAITH_CONTEXT_SETUP_LOGGED:
        return
    _FAITH_CONTEXT_SETUP_LOGGED = True
    primary = _resolve_faith_rag_primary_dir()
    gold = _resolve_faith_rag_gold_dir()
    mode = (context_mode or "").strip().lower()
    if mode == "stored":
        print(
            "[faithfulness] context_mode=stored → judging against CSV retrieved_context only. "
            "For better scores attach fatinshadab/rag-index and set "
            "GP_FAITH_CONTEXT=reretrieve_or_stored.",
            flush=True,
        )
        return
    if mode != "stored" and not primary and not gold:
        print(
            "[faithfulness] WARNING: no RAG index found — falling back to stored CSV context. "
            "On Kaggle attach: fatinshadab/rag-index (+ rag-index-gold for PubMedQA), then:\n"
            "  os.environ['GP_FAITH_CONTEXT']='reretrieve_or_stored'\n"
            f"  Expected: {_KAGGLE_RAG_FAISS_PRIMARY!r}, {_KAGGLE_RAG_FAISS_GOLD!r}",
            flush=True,
        )
    elif primary or gold:
        print(
            f"[faithfulness] RAG indexes ready:\n"
            f"  primary={primary or '—'} ({_describe_rag_index_dir(primary)})\n"
            f"  gold={gold or '—'} ({_describe_rag_index_dir(gold)})",
            flush=True,
        )


def _faith_context_needs_reretrieve(context_mode: str) -> bool:
    return (context_mode or "").strip().lower() in (
        "reretrieve",
        "gold",
        "reretrieve_or_stored",
    )


def _rag_helpers_available() -> bool:
    work = "/kaggle/working" if os.path.isdir("/kaggle/working") else _script_dir()
    if os.path.isfile(os.path.join(work, "real_model_runner.py")):
        return True
    return bool(_find_real_model_runner_root())


def _faith_rag_retrieval_available() -> bool:
    return _rag_helpers_available() or _inline_rag_retrieval_available()


def _require_rag_helpers_if_needed(context_mode: str) -> None:
    """Fail fast when re-retrieval is configured but no retrieval backend is available."""
    if not _faith_context_needs_reretrieve(context_mode):
        return
    primary = _resolve_faith_rag_primary_dir()
    gold = _resolve_faith_rag_gold_dir()
    if not primary and not gold:
        print(
            "[faithfulness] No RAG indexes on disk — using stored CSV context "
            "(reretrieve_or_stored / stored fallback).",
            flush=True,
        )
        return
    if os.path.isdir("/kaggle") and not _inline_rag_deps_ok():
        _ensure_rag_retrieval_deps()
    if not _faith_rag_retrieval_available():
        if primary or gold:
            _ensure_rag_retrieval_deps()
    if _faith_rag_retrieval_available():
        if not _rag_helpers_available() and _inline_rag_retrieval_available():
            _log_inline_rag_backend_once()
        return
    if primary and not _inline_rag_deps_ok():
        msg = (
            "[faithfulness] ERROR: RAG indexes are attached but faiss-cpu / "
            "sentence-transformers are not installed. Run in a cell before faithfulness:\n"
            "  !pip install -q faiss-cpu sentence-transformers\n"
            "Then Kernel → Restart Session and re-run. "
            "(Auto-pip failed or GP_FAITH_NO_AUTO_PIP=1.)"
        )
    else:
        p = os.environ.get("GP_FAITH_RAG_INDEX_DIR", "").strip()
        g = os.environ.get("GP_FAITH_RAG_GOLD_DIR", "").strip()
        msg = (
            "[faithfulness] ERROR: context_mode requires RAG re-retrieval but no backend is ready. "
            "Notebook → Add Data → rag-index + rag-index-gold "
            "(chunks.jsonl + index.faiss).\n"
            f"  GP_FAITH_RAG_INDEX_DIR={p!r} GP_FAITH_RAG_GOLD_DIR={g!r}\n"
            "  Or: !pip install -q faiss-cpu sentence-transformers  then restart kernel.\n"
            "  CSV-only: os.environ['GP_FAITH_CONTEXT']='stored'\n"
            "  Weak fallback: GP_FAITH_ALLOW_STORED_FALLBACK=1"
        )
    if os.environ.get("GP_FAITH_ALLOW_STORED_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("GP_FAITH_GAP_FILL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        print(msg.replace("ERROR:", "WARNING:"), flush=True)
        return
    raise ImportError(msg)


def _init_faithfulness_rag(context_mode: str) -> None:
    if context_mode == "stored":
        return
    primary = _resolve_faith_rag_primary_dir()
    gold = _resolve_faith_rag_gold_dir()
    if not primary and not gold:
        print(
            "[faithfulness] No rag_index / rag_index_gold found — using stored CSV context only.",
            flush=True,
        )
        return
    _require_rag_helpers_if_needed(context_mode)
    primary = _resolve_faith_rag_primary_dir()
    gold = _resolve_faith_rag_gold_dir()
    print(
        f"[faithfulness] RAG context_mode={context_mode}; "
        f"primary_index={primary or 'missing'}; gold_index={gold or 'missing'}",
        flush=True,
    )
    if not _FAITH_CONTEXT_SETUP_LOGGED:
        _log_pinned_rag_paths(primary, gold)
    if not primary and not gold:
        print(
            "[faithfulness] No rag_index / rag_index_gold found — using stored CSV context only.",
            flush=True,
        )
        return
    for idx_dir in (primary, gold):
        if not idx_dir:
            continue
        try:
            _prewarm_faith_rag(idx_dir)
        except Exception as exc:
            print(f"[faithfulness] RAG prewarm warning ({idx_dir}): {exc}", flush=True)


def _is_pubmedqa_row(row: Dict[str, Any]) -> bool:
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    b = str(row.get("benchmark") or src.get("benchmark") or "").strip().lower()
    return b == "pubmedqa" or str(row.get("question_id") or "").startswith("pubmedqa_")


def _is_mcq_benchmark_row(row: Dict[str, Any]) -> bool:
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    b = str(row.get("benchmark") or src.get("benchmark") or "").strip().lower()
    return b in ("medqa", "mmlu_med", "mmlu-med", "mmlu")


def _retrieval_query_from_row(row: Dict[str, Any], question: str) -> str:
    """Enrich retrieval query with MCQ option text (MedQA/MMLU) for better lexical match."""
    q = (question or "").strip()
    if not q or not _is_mcq_benchmark_row(row):
        return q
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    cj = str(src.get("choices_json") or "").strip()
    if not cj:
        return q
    try:
        data = json.loads(cj)
    except (json.JSONDecodeError, TypeError):
        return q
    parts: List[str] = [q]
    if isinstance(data, dict):
        for letter in sorted(data.keys()):
            parts.append(f"{letter}. {data[letter]}")
    elif isinstance(data, list):
        for item in data[:6]:
            parts.append(str(item).strip())
    opt_blob = " ".join(parts[1:])
    if opt_blob:
        return f"{q}\n\nAnswer choices:\n{opt_blob[:3500]}"
    return q


def _reretrieve_evidence_block(row: Dict[str, Any], *, use_gold: bool) -> Tuple[str, str]:
    """Return (evidence_block, source_label) from local FAISS/BM25 indexes."""
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    q = str(row.get("question") or src.get("question") or "").strip()
    if not q:
        return "", "reretrieve_empty_query"
    query = _retrieval_query_from_row(row, q)
    gold_dir = _resolve_faith_rag_gold_dir()
    primary_dir = _resolve_faith_rag_primary_dir()
    try:
        if use_gold and _is_pubmedqa_row(row) and gold_dir:
            with _temporary_rag_corpus(gold_dir):
                with _pubmedqa_gold_retrieval_tuning():
                    block = _prefetch_rag_for_query(query)
            label = "reretrieve_gold"
        elif primary_dir:
            with _temporary_rag_corpus(primary_dir):
                with _faithfulness_primary_retrieval_tuning():
                    block = _prefetch_rag_for_query(query)
                    if _is_mcq_benchmark_row(row) and os.environ.get(
                        "GP_FAITH_OPTION_RETRIEVE", "1"
                    ).strip().lower() in ("1", "true", "yes"):
                        opt_q = _option_focused_retrieval_query(row)
                        if opt_q and len(opt_q) > 20:
                            block2 = _prefetch_rag_for_query(opt_q)
                            block = _merge_evidence_text(block, block2)
            label = "reretrieve_primary"
        else:
            return "", "reretrieve_no_index"
    except Exception as exc:
        return "", f"reretrieve_failed:{str(exc)[:80]}"
    block = str(block or "").strip()
    return block, label if block else f"{label}_empty"


def _resolve_scoring_context(
    row: Dict[str, Any],
    context_mode: str,
    *,
    strip_prompt: bool,
) -> Tuple[str, str]:
    """Pick CONTEXT for the judge; return (context_text, source_label)."""
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    if not _rag_flag_true(src) and not _rag_flag_true(row):
        norag = _norag_grounding_context(row)
        if norag:
            ctx = _prepare_context_for_scoring(norag, strip_prompt=strip_prompt)
            return ctx, "norag_grounding"
    stored_raw = str(row.get("context_stored") or row.get("context") or "").strip()
    mode = (context_mode or "stored").strip().lower()
    if mode == "stored":
        ctx = _prepare_context_for_scoring(stored_raw, strip_prompt=strip_prompt)
        return ctx, "stored"

    use_gold = mode == "gold" or (
        mode == "reretrieve_or_stored" and _is_pubmedqa_row(row)
    )
    fresh_raw, label = _reretrieve_evidence_block(row, use_gold=use_gold)
    fresh = _prepare_context_for_scoring(fresh_raw, strip_prompt=strip_prompt)

    if mode == "reretrieve":
        return (fresh, label) if fresh else ("", label)
    if mode == "gold":
        if fresh:
            return fresh, label
        ctx = _prepare_context_for_scoring(stored_raw, strip_prompt=strip_prompt)
        return ctx, "stored_fallback"
    # reretrieve_or_stored
    return _pick_reretrieve_or_stored_context(
        row,
        fresh_raw=fresh_raw,
        fresh_label=label,
        stored_raw=stored_raw,
        strip_prompt=strip_prompt,
    )


def _fresh_has_evidence_chunks(context: str) -> bool:
    """True when re-retrieved context looks like real ranked evidence chunks."""
    t = (context or "").strip()
    if len(t) < 120:
        return False
    return bool(re.search(r"Chunk \d+ \(", t) or re.search(r"\[\d+\]", t))


def _stored_clearly_better(fs: float, ss: float) -> bool:
    """Stored CSV context must clearly beat re-retrieval to replace it."""
    margin = float(os.environ.get("GP_FAITH_STORED_MARGIN", "1.45") or "1.45")
    min_stored = float(os.environ.get("GP_FAITH_MIN_STORED_SCORE", "0.20") or "0.20")
    return ss >= min_stored and ss >= fs * margin and ss > fs + 0.04


def _merge_context_worthwhile(fs: float, ss: float) -> bool:
    """Merge only when both contexts contribute (avoid diluting good stored-only rows)."""
    min_both = float(os.environ.get("GP_FAITH_MERGE_MIN_SCORE", "0.06") or "0.06")
    max_gap = float(os.environ.get("GP_FAITH_MERGE_MAX_GAP", "0.10") or "0.10")
    return fs >= min_both and ss >= min_both and abs(fs - ss) <= max_gap


def _pick_reretrieve_or_stored_context(
    row: Dict[str, Any],
    *,
    fresh_raw: str,
    fresh_label: str,
    stored_raw: str,
    strip_prompt: bool,
) -> Tuple[str, str]:
    """
    Prefer re-retrieved evidence; PubMedQA gold is never overridden.
    MedQA/MMLU: merge only when both sources contribute; else pick the stronger one.
    """
    fresh = _prepare_context_for_scoring(fresh_raw, strip_prompt=strip_prompt)
    stored = _prepare_context_for_scoring(stored_raw, strip_prompt=strip_prompt)
    answer = str(row.get("answer") or "").strip()
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    question = str(row.get("question") or src.get("question") or "").strip()

    if not fresh or len(fresh) < 80:
        return (stored, "stored") if stored else (fresh, fresh_label)
    if not stored or len(stored) < 80:
        return fresh, fresh_label

    if fresh_label == "reretrieve_gold" or _is_pubmedqa_row(row):
        return fresh, fresh_label

    if os.environ.get("GP_FAITH_BEST_CONTEXT", "1").strip().lower() in ("0", "false", "no"):
        return fresh, fresh_label

    fs = _context_support_score(fresh, answer, question)
    ss = _context_support_score(stored, answer, question)
    max_c = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "6500") or "6500")
    blend_medqa = os.environ.get("GP_FAITH_MEDQA_BLEND", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if _stored_clearly_better(fs, ss):
        return stored, "stored_better_support"

    # MedQA/MMLU: default to re-retrieve + stored blend (better than stored-only).
    if (
        blend_medqa
        and _is_mcq_benchmark_row(row)
        and not _is_pubmedqa_row(row)
        and _fresh_has_evidence_chunks(fresh)
        and len(stored) >= 80
    ):
        blended = _blend_reretrieve_stored_context(fresh, stored, max_chars=max_c)
        if len(blended) >= 80:
            return blended, "reretrieve_plus_stored"

    merge_medqa = os.environ.get("GP_FAITH_MERGE_MEDQA", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if (
        merge_medqa
        and _is_mcq_benchmark_row(row)
        and not _is_pubmedqa_row(row)
        and _fresh_has_evidence_chunks(fresh)
        and len(stored) >= 80
        and _merge_context_worthwhile(fs, ss)
    ):
        merged = _blend_reretrieve_stored_context(
            fresh, stored, max_chars=max_c, fresh_frac=0.50
        )
        if len(merged) >= 80:
            return merged, "merged_reretrieve_stored"

    if fresh_label.startswith("reretrieve") and _fresh_has_evidence_chunks(fresh):
        if fs >= ss * 0.88:
            return fresh, fresh_label

    slight_margin = float(os.environ.get("GP_FAITH_STORED_SLIGHT_MARGIN", "1.15") or "1.15")
    if ss >= fs * slight_margin and ss >= 0.12:
        return stored, "stored_better_support"

    if os.environ.get("GP_FAITH_MERGE_CONTEXT", "0").strip().lower() in ("1", "true", "yes"):
        merged = _blend_reretrieve_stored_context(fresh, stored, max_chars=max_c)
        if len(merged) >= 80:
            return merged, "merged_reretrieve_stored"

    return fresh, fresh_label


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _ensure_bitsandbytes() -> bool:
    if _bitsandbytes_available():
        return True
    if os.environ.get("GP_FAITH_NO_AUTO_PIP", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        _pip_install_packages(["bitsandbytes"])
    except Exception as exc:
        print(f"[faithfulness] bitsandbytes install failed: {exc}", flush=True)
        return False
    return _bitsandbytes_available()


def _clear_local_hf_gpu_cache(model_id: str = "") -> None:
    """Drop cached HF models so OOM fallback can load a smaller judge."""
    import gc

    global _LOCAL_HF_MODELS
    if model_id:
        mid = model_id.strip()
        if mid in _LOCAL_HF_MODELS:
            pair = _LOCAL_HF_MODELS.pop(mid)
            try:
                del pair
            except Exception:
                pass
    else:
        _LOCAL_HF_MODELS.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _default_use_4bit() -> bool:
    env = os.environ.get("GP_FAITH_USE_4BIT", "").strip().lower()
    if env in ("0", "false", "no", "n"):
        return False
    if env in ("1", "true", "yes", "y"):
        return True
        return False


def _use_4bit_for_judge(model_id: str) -> bool:
    """4-bit on T4 for 7B+ judges when bitsandbytes is available."""
    if _default_use_4bit():
        return _bitsandbytes_available()
    if os.environ.get("GP_FAITH_USE_4BIT", "").strip().lower() in ("0", "false", "no", "n"):
        return False
    if not os.path.isdir("/kaggle"):
        return False
    m = (model_id or "").lower()
    if any(x in m for x in ("7b", "8b", "14b", "32b", "70b", "72b")):
        return _bitsandbytes_available()
    return False


def _try_load_hf_token() -> bool:
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return True
    try:
        from huggingface_hub import get_token

        t = get_token()
        if t and str(t).strip():
            os.environ.setdefault("HF_TOKEN", str(t).strip())
            return True
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient

        c = UserSecretsClient()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token"):
            try:
                v = c.get_secret(name)
                if v and str(v).strip():
                    os.environ["HF_TOKEN"] = str(v).strip()
                    print(f"[faithfulness] Loaded HF_TOKEN from Kaggle Secrets ({name}).", flush=True)
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n...[truncated]..."


def _strip_qwen_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks so JSON parsers see the answer."""
    t = (text or "").strip()
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    if close_tag in t:
        t = t.split(close_tag, 1)[-1].strip()
    t = re.sub(
        re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag),
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    return t


def _gpu_device_map() -> Any:
    """Keep the full model on GPU (avoid T4 CPU offload with Qwen3-8B)."""
    import torch

    if torch.cuda.is_available():
        return {"": 0}
    return None


def _load_local_hf_model(model_id: str, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        kwargs["token"] = token

    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    load_kw: Dict[str, Any] = {**kwargs}
    want_4bit = use_4bit and torch.cuda.is_available() and _bitsandbytes_available()
    if use_4bit and not want_4bit and not _bitsandbytes_available():
        print(
            "[faithfulness] 4-bit requested but bitsandbytes missing; loading fp16.",
            flush=True,
        )
    elif not use_4bit and _use_4bit_for_judge(model_id):
        want_4bit = True
        print(
            f"[faithfulness] T4: loading {model_id!r} in 4-bit (set GP_FAITH_USE_4BIT=0 to disable).",
            flush=True,
        )
    if use_4bit and not want_4bit:
        print(
            "[faithfulness] Loading fp16 on GPU (bitsandbytes unavailable or disabled).",
            flush=True,
        )
    if want_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kw["device_map"] = "auto"
        except Exception:
            load_kw["dtype"] = torch.float16
            load_kw["device_map"] = "auto"
    else:
        load_kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        dm = _gpu_device_map()
        if dm is not None:
            load_kw["device_map"] = dm

    def _load(**kw: Any):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except TypeError:
            if "dtype" in kw:
                kw = {**kw, "torch_dtype": kw.pop("dtype")}
            return AutoModelForCausalLM.from_pretrained(model_id, **kw)

    try:
        model = _load(**load_kw)
    except Exception as exc:
        err = str(exc).lower()
        if "out of memory" in err and not want_4bit and _ensure_bitsandbytes():
            print(
                f"[faithfulness] OOM loading {model_id!r}; clearing GPU and retrying 4-bit.",
                flush=True,
            )
            _clear_local_hf_gpu_cache(model_id)
            return _load_local_hf_model(model_id, use_4bit=True)
        if model_id != _FALLBACK_FAITHFULNESS_MODEL and (
            "out of memory" in err or "cuda" in err or "not found" in err
        ):
            print(
                f"[faithfulness] Failed to load {model_id!r} ({exc}); "
                f"clearing GPU and retrying {_FALLBACK_FAITHFULNESS_MODEL!r} (4-bit if needed).",
                flush=True,
            )
            _clear_local_hf_gpu_cache(model_id)
            fb_4bit = _use_4bit_for_judge(_FALLBACK_FAITHFULNESS_MODEL) or os.path.isdir(
                "/kaggle"
            )
            return _load_local_hf_model(
                _FALLBACK_FAITHFULNESS_MODEL, use_4bit=fb_4bit
            )
        if "bitsandbytes" in err:
            print("[faithfulness] 4-bit load failed; retrying fp16 on GPU.", flush=True)
            load_kw = {**kwargs}
            load_kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            load_kw["device_map"] = "auto" if torch.cuda.is_available() else None
            load_kw.pop("quantization_config", None)
            model = _load(**load_kw)
        else:
            raise
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        import torch

        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(f"[faithfulness] Local judge loaded: {model_id!r} on {dev}", flush=True)
    return model, tok


def _hf_generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    formatted = prompt
    try:
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("no chat_template")
        tpl_kw: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **tpl_kw,
            )
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, **tpl_kw)
    except Exception:
        formatted = prompt

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    gen_cfg = getattr(model, "generation_config", None)
    saved_temp = getattr(gen_cfg, "temperature", None) if gen_cfg is not None else None
    saved_top_p = getattr(gen_cfg, "top_p", None) if gen_cfg is not None else None
    if gen_cfg is not None:
        try:
            gen_cfg.temperature = None
            gen_cfg.top_p = None
            gen_cfg.top_k = None
        except Exception:
            pass
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max(64, min(768, max_new_tokens)),
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
        )
    if gen_cfg is not None:
        try:
            if saved_temp is not None:
                gen_cfg.temperature = saved_temp
            if saved_top_p is not None:
                gen_cfg.top_p = saved_top_p
        except Exception:
            pass
    new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
    return _strip_qwen_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


class _QwenFaithfulnessCore:
    """Shared HF generate logic for DeepEval judge."""

    def __init__(
        self,
        model_id: str,
        *,
        use_4bit: bool = False,
        max_new_tokens: int = 512,
    ):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load_model(self):
        if self._model is None:
            self._model, self._tokenizer = _load_local_hf_model(self.model_id, self.use_4bit)
        return self._model

    def get_model_name(self) -> str:
        return self.model_id

    def generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is not None:
            try:
                schema_hint = json.dumps(schema.model_json_schema(), indent=2)
            except Exception:
                schema_hint = str(schema)
            prompt = (
                f"{prompt}\n\n"
                "Respond with valid JSON only (no markdown, no thinking tags) matching:\n"
                f"{schema_hint}"
            )
        raw = _hf_generate_text(self.load_model(), self._tokenizer, prompt, self.max_new_tokens)
        if schema is None:
            return raw
        parsed = _extract_json_object(raw)
        if parsed is not None:
            try:
                return schema.model_validate(parsed)
            except Exception:
                pass
        try:
            return schema.model_validate_json(_strip_qwen_thinking(raw))
        except Exception as exc:
            raise ValueError(f"DeepEval JSON parse failed: {raw[:300]!r}") from exc

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)


def _make_qwen_deepeval_llm(model_id: str) -> Any:
    from deepeval.models.base_model import DeepEvalBaseLLM

    class _QwenLLM(DeepEvalBaseLLM):
        def __init__(self):
            self._inner = _QwenFaithfulnessCore(
                model_id,
                use_4bit=_default_use_4bit(),
            )

        def load_model(self):
            return self._inner.load_model()

        def get_model_name(self) -> str:
            return self._inner.get_model_name()

        def generate(self, prompt: str, schema: Any = None) -> Any:
            return self._inner.generate(prompt, schema)

        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            return self._inner.a_generate(prompt, schema)

    return _QwenLLM()


def _get_deepeval_metric(model_id: str):
    if model_id not in _DEEPEVAL_METRICS:
        from deepeval.metrics import FaithfulnessMetric

        llm = _make_qwen_deepeval_llm(model_id)
        try:
            _DEEPEVAL_METRICS[model_id] = FaithfulnessMetric(
                model=llm,
                threshold=0.5,
                async_mode=False,
                verbose_mode=False,
            )
        except TypeError:
            _DEEPEVAL_METRICS[model_id] = FaithfulnessMetric(
                model=llm,
                threshold=0.5,
                async_mode=False,
            )
    return _DEEPEVAL_METRICS[model_id]


def _log_backend(backend: str, model_id: str = "") -> None:
    if backend == "openrouter":
        print("[faithfulness] Backend: OpenRouter API.", flush=True)
        return
    mid = model_id or _resolve_faithfulness_model()
    label = "local HF judge (1 call/row)" if backend == "local_hf" else "DeepEval + local HF"
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(
            f"[faithfulness] Backend: {label} ({mid or 'default'}) "
            f"on cuda={cuda}, device={dev}.",
            flush=True,
        )
    except Exception:
        print(f"[faithfulness] Backend: DeepEval + local HF ({model_id}).", flush=True)


SYSTEM_PROMPT = """You are a strict medical NLP evaluator. Your job is to judge whether a model ANSWER is faithful to a given CONTEXT (retrieved evidence only).

Rules:
- Score FAITHFULNESS from 0 to 100 (integer): how much of the ANSWER is directly supported by or entailed by the CONTEXT.
- If the ANSWER adds facts not in the CONTEXT, lower the score.
- Clinical hedging consistent with CONTEXT is OK; contradictions or unsupported claims lower the score.
- Output ONLY valid JSON, no markdown, on a single line: {"faithfulness": <int 0-100>, "note": "<short string>"}
"""

SYSTEM_PROMPT_MCQ = """You are a medical NLP evaluator for multiple-choice (MCQ) RAG answers.

Score whether the SELECTED OPTION in ANSWER (letter + option text) is reasonably supported by CONTEXT (retrieved evidence only).

Rules:
- Judge medical entailment/consistency, NOT whether the exact letter or wording appears in CONTEXT.
- If CONTEXT discusses the same disease, drug class, anatomy, or mechanism as the selected option, score at least 45 unless CONTEXT clearly contradicts the option.
- 100 = CONTEXT clearly supports or entails the selected option as a reasonable answer
- 60-80 = CONTEXT is related and provides a reasonable clinical basis for this option
- 40-55 = CONTEXT is partially related; indirect or incomplete support
- 25-38 = CONTEXT is tangential; weak basis for this option
- 0-20 = CONTEXT is irrelevant, contradicts, or gives no basis for this option
- Avoid defaulting to 20 or 40; use granular scores (e.g. 28 tangential, 47 partial, 68 reasonable).
- Examples: same drug class mentioned → ~65; same organ system only → ~32; unrelated topic → ~8.
- Output ONLY valid JSON on one line: {"faithfulness": <int 0-100>, "note": "<short string>"}
"""


def _extract_evidence_only(context: str) -> str:
    """Keep retrieved evidence chunks; drop RAG instruction boilerplate."""
    t = (context or "").strip()
    if not t:
        return t
    for marker in (
        "Retrieved Medical Evidence:",
        "Retrieved Medical Evidence",
        "Medical Evidence:",
    ):
        idx = t.find(marker)
        if idx >= 0:
            return t[idx:].strip()
    m = re.search(r"\[\d+\]", t)
    if m:
        return t[m.start() :].strip()
    low = t.lower()
    if low.startswith("you are a biomedical") or low.startswith("you are a medical"):
        parts = re.split(r"\n\s*\n", t, maxsplit=2)
        if len(parts) >= 2 and re.search(r"\[\d+\]", parts[-1]):
            return parts[-1].strip()
    return t


def _option_text_from_choices(choices_json: str, letter: str) -> str:
    letter = (letter or "").strip().upper()[:1]
    if not letter or not choices_json:
        return ""
    try:
        data = json.loads(choices_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(data, dict):
        if letter in data:
            return str(data[letter]).strip()
        for k, v in data.items():
            if str(k).strip().upper()[:1] == letter:
                return str(v).strip()
        return ""
    if isinstance(data, list) and data:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(data):
            item = str(data[idx]).strip()
            pat = re.compile(rf"^\s*{re.escape(letter)}[\.\)\:\-]\s*", re.IGNORECASE)
            if pat.match(item):
                return pat.sub("", item, count=1).strip()
            return item
        pat = re.compile(rf"^\s*{re.escape(letter)}[\.\)\:\-]\s*", re.IGNORECASE)
        for item in data:
            s = str(item).strip()
            if pat.match(s):
                return pat.sub("", s, count=1).strip()
    return ""


def _build_faithfulness_answer(row: Dict[str, Any], *, mcq_only: bool) -> str:
    """Answer text sent to the judge (MCQ option vs full model response)."""
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    parsed = str(
        src.get("parsed_prediction") or row.get("parsed_prediction") or ""
    ).strip()
    if mcq_only and parsed:
        letter = parsed.upper()[:1]
        if letter.isalpha() and len(parsed) <= 3:
            opt = _option_text_from_choices(str(src.get("choices_json") or ""), letter)
            if not opt:
                opt = str(src.get("prediction_text") or "").strip()
            if opt:
                return f"Selected MCQ option: {letter}. Option text: {opt}"
            return f"Selected MCQ option: {letter}."
    return str(
        row.get("answer")
        or src.get("raw_response")
        or src.get("model_answer")
        or ""
    ).strip()


def _prepare_context_for_scoring(context: str, *, strip_prompt: bool) -> str:
    ctx = (context or "").strip()
    if strip_prompt:
        ctx = _extract_evidence_only(ctx)
    return ctx


def _system_prompt_for_mode(mcq_only: bool) -> str:
    return SYSTEM_PROMPT_MCQ if mcq_only else SYSTEM_PROMPT


def _retrieval_fields_from_source(row: Dict[str, Any]) -> Dict[str, Any]:
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    out: Dict[str, Any] = {}
    for key, out_key in (
        ("evidence_token_overlap", "evidence_token_overlap"),
        ("gold_chunk_found", "gold_chunk_found"),
        ("mcq_correct", "mcq_correct"),
        ("label_correct", "label_correct"),
        ("parsed_prediction", "parsed_prediction"),
        ("recall_at_1", "recall_at_1"),
        ("recall_at_5", "recall_at_5"),
        ("rag_context_used", "rag_context_used"),
    ):
        if key in src and src.get(key) not in ("", None):
            out[out_key] = src.get(key)
    return out


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_openrouter(
    api_key: str,
    model: str,
    user_content: str,
    timeout: int = 120,
    max_retries: int = 5,
    system_prompt: str = "",
) -> str:
    if requests is None:
        raise ImportError("pip install requests")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://kaggle.com"),
        "X-Title": "GreenPaper-FaithfulnessEval",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code == 429:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(min(30, 1.5 ** attempt))
    raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")


def _get_local_hf_model(model_id: str):
    if model_id not in _LOCAL_HF_MODELS:
        _try_load_hf_token()
        if _use_4bit_for_judge(model_id):
            _ensure_bitsandbytes()
        _LOCAL_HF_MODELS[model_id] = _load_local_hf_model(
            model_id, use_4bit=_use_4bit_for_judge(model_id)
        )
    return _LOCAL_HF_MODELS[model_id]


def score_one_local_hf(
    question: str,
    context: str,
    answer: str,
    *,
    model_id: str = "",
    max_ctx_chars: int = 6000,
    mcq_only: bool = False,
) -> Tuple[float, str]:
    """Single-call local HF faithfulness judge (1 GPU call per row; reliable on T4)."""
    mid = _resolve_faithfulness_model(model_id)
    model, tok = _get_local_hf_model(mid)
    ctx = _truncate(context, max_ctx_chars)
    ans = _truncate(answer, max_ctx_chars)
    sys_p = _system_prompt_for_mode(mcq_only)
    ans_label = "SELECTED OPTION to score" if mcq_only else "ANSWER to score"
    user_content = (
        f"QUESTION (optional context for you, not extra evidence):\n{question}\n\n"
        f"CONTEXT (only source of truth; may include multiple evidence chunks):\n{ctx}\n\n"
        f"{ans_label}:\n{ans}\n\n"
        "Score how well CONTEXT supports the selected option (partial clinical relevance counts).\n"
        'Return JSON only: {"faithfulness": <0-100 int>, "note": "<short string>"}'
    )
    prompt = f"{sys_p}\n\n{user_content}"
    raw = _hf_generate_text(model, tok, prompt, max_new_tokens=256)
    parsed = _extract_json_object(raw)
    if not parsed or "faithfulness" not in parsed:
        return float("nan"), raw[:500]
    try:
        f = float(parsed["faithfulness"])
        f = max(0.0, min(100.0, f))
        note = str(parsed.get("note", ""))
        if mcq_only:
            f, note = _calibrate_mcq_faithfulness_score(f, note, ctx, ans, question)
        return f, note
    except (TypeError, ValueError):
        return float("nan"), raw[:500]


def score_one_deepeval(
    question: str,
    context: str,
    answer: str,
    *,
    metric=None,
    model_id: str = "",
    max_ctx_chars: int = 6000,
    mcq_only: bool = False,
) -> Tuple[float, str]:
    """DeepEval faithfulness 0--1 scaled to 0--100."""
    from deepeval.metrics import FaithfulnessMetric  # noqa: F401
    from deepeval.test_case import LLMTestCase

    ctx = _truncate(context, max_ctx_chars)
    ans = _truncate(answer, max_ctx_chars)
    ctx_chunks = [ctx] if ctx else []
    if not ctx_chunks:
        return float("nan"), "empty_context"
    if not ans:
        return float("nan"), "empty_answer"

    mid = _resolve_faithfulness_model(model_id)
    if metric is None:
        metric = _get_deepeval_metric(mid)

    test_case = LLMTestCase(
        input=question or "Clinical question",
        actual_output=ans,
        retrieval_context=ctx_chunks,
    )
    try:
        metric.measure(test_case)
        score = float(metric.score) if metric.score is not None else float("nan")
        if score == score and score <= 1.0:
            score *= 100.0
        note = str(getattr(metric, "reason", "") or getattr(metric, "error", "") or "")[:500]
        if score == score:
            return max(0.0, min(100.0, score)), note
        # DeepEval + small local models often fail JSON on multi-step prompts — fallback.
        fb, fb_note = score_one_local_hf(
            question,
            context,
            answer,
            model_id=mid,
            max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
        )
        if fb == fb:
            return fb, f"deepeval_fallback: {fb_note}"[:500]
        return float("nan"), note or fb_note
    except Exception as ex:
        fb, fb_note = score_one_local_hf(
            question,
            context,
            answer,
            model_id=mid,
            max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
        )
        if fb == fb:
            return fb, f"deepeval_err_fallback: {fb_note}"[:500]
        return float("nan"), str(ex)[:500]


def score_one_openrouter(
    api_key: str,
    model: str,
    question: str,
    context: str,
    answer: str,
    *,
    max_ctx_chars: int = 6000,
    mcq_only: bool = False,
) -> Tuple[float, str]:
    ctx = _truncate(context, max_ctx_chars)
    ans = _truncate(answer, max_ctx_chars)
    ans_label = "SELECTED OPTION to score" if mcq_only else "ANSWER to score"
    user_content = (
        f"QUESTION (optional context for you, not extra evidence):\n{question}\n\n"
        f"CONTEXT (only source of truth):\n{ctx}\n\n"
        f"{ans_label}:\n{ans}\n\n"
        "Return JSON only: {\"faithfulness\": <0-100 int>, \"note\": \"...\"}"
    )
    raw = call_openrouter(api_key, model, user_content, system_prompt=_system_prompt_for_mode(mcq_only))
    parsed = _extract_json_object(raw)
    if not parsed or "faithfulness" not in parsed:
        return float("nan"), raw[:500]
    try:
        f = float(parsed["faithfulness"])
        f = max(0.0, min(100.0, f))
        return f, str(parsed.get("note", ""))
    except (TypeError, ValueError):
        return float("nan"), raw[:500]


def score_one(
    api_key: str,
    model: str,
    question: str,
    context: str,
    answer: str,
    *,
    backend: str = "openrouter",
    deepeval_metric=None,
    model_id: str = "",
    max_ctx_chars: int = 6000,
    mcq_only: bool = False,
) -> Tuple[float, str]:
    b = (backend or "openrouter").strip().lower()
    if b == "local_hf":
        return score_one_local_hf(
            question,
            context,
            answer,
            model_id=model_id,
            max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
        )
    if b == "deepeval":
        return score_one_deepeval(
            question,
            context,
            answer,
            metric=deepeval_metric,
            model_id=model_id,
            max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
        )
    if b == "auto":
        return score_one_local_hf(
                question,
                context,
                answer,
                model_id=model_id,
                max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
            )
    return score_one_openrouter(
        api_key,
        model,
        question,
        context,
        answer,
        max_ctx_chars=max_ctx_chars,
        mcq_only=mcq_only,
    )


def _rag_flag_true(row: Dict[str, Any]) -> bool:
    v = row.get("rag_flag")
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("true", "1", "yes", "t")


def _row_key(row: Dict[str, Any]) -> str:
    mname = str(row.get("model_name") or "")
    if not mname:
        mname = f"{row.get('model_key') or ''}_{'rag' if _rag_flag_true(row) else 'norag'}"
    return "|".join(
        [
            str(row.get("run_folder") or ""),
            str(row.get("source_file") or ""),
            str(row.get("benchmark") or ""),
            str(row.get("question_id") or row.get("id") or ""),
            mname,
        ]
    )


def _row_key_relaxed(row: Dict[str, Any]) -> str:
    """Match prior batches across accounts when ``run_folder`` / ``source_file`` differ."""
    mname = str(row.get("model_name") or "")
    if not mname:
        mname = f"{row.get('model_key') or ''}_{'rag' if _rag_flag_true(row) else 'norag'}"
    return "|".join(
        [
            str(row.get("benchmark") or ""),
            str(row.get("question_id") or row.get("id") or ""),
            mname,
        ]
    )


def _prior_match_strict_only() -> bool:
    """When true, only exact ``_row_key`` matches count as already scored (gap-fill mode)."""
    return os.environ.get("GP_FAITH_PRIOR_MATCH_STRICT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def _work_row_keys_covered_by_prior(
    work_rows: List[Dict[str, Any]],
    scored_records: Dict[str, Dict[str, Any]],
) -> Tuple[set, Dict[str, int]]:
    """
    Work-row ``_row_key`` values already scored in prior batch CSVs.

    Uses strict key first, then relaxed ``benchmark|question_id|model_name`` fallback
    unless ``GP_FAITH_PRIOR_MATCH_STRICT=1``.
    """
    strict_prior = set(scored_records.keys())
    relaxed_prior: Dict[str, str] = {}
    if not _prior_match_strict_only():
        for sk, row in scored_records.items():
            relaxed_prior[_row_key_relaxed(row)] = sk

    covered: set = set()
    stats = {"strict": 0, "relaxed": 0, "prior_keys_in_files": len(scored_records)}
    for wr in work_rows:
        wk = _row_key(wr)
        if wk in strict_prior:
            covered.add(wk)
            stats["strict"] += 1
        elif not _prior_match_strict_only():
            rk = _row_key_relaxed(wr)
            if rk in relaxed_prior:
                covered.add(wk)
                stats["relaxed"] += 1
    return covered, stats


def _prior_record_for_work_row(
    work_row: Dict[str, Any],
    scored_records: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    k = _row_key(work_row)
    if k in scored_records:
        return scored_records[k]
    rk = _row_key_relaxed(work_row)
    for row in scored_records.values():
        if _row_key_relaxed(row) == rk:
            return row
    return None


def _user_start_batch_1based(start_batch_param: int = -1) -> int:
    """1-indexed first batch: ``GP_FAITH_START_BATCH`` env, then arg, then account default."""
    env_raw = os.environ.get("GP_FAITH_START_BATCH", "").strip()
    if env_raw:
        try:
            return max(1, int(env_raw))
        except ValueError:
            pass
    if start_batch_param >= 1:
        return start_batch_param
    return _default_start_batch()


def _norag_grounding_context(row: Dict[str, Any]) -> str:
    """
    Evidence available without retrieval (for comparable NoRAG faithfulness).

    PubMedQA: task abstract in ``context``. MCQ: question stem plus answer choices.
    """
    src = row.get("_source_row") if isinstance(row.get("_source_row"), dict) else row
    if _is_pubmedqa_row(row):
        return str(src.get("context") or src.get("reference_text") or "").strip()
    if _is_mcq_benchmark_row(row):
        q = str(row.get("question") or src.get("question") or "").strip()
        cj = str(src.get("choices_json") or "").strip()
        parts: List[str] = [q] if q else []
        if cj:
            try:
                data = json.loads(cj)
                if isinstance(data, dict):
                    for letter in sorted(data.keys()):
                        parts.append(f"{letter}. {data[letter]}")
                elif isinstance(data, list):
                    for i, item in enumerate(data[:6]):
                        parts.append(f"{chr(ord('A') + i)}. {item}")
            except (json.JSONDecodeError, TypeError):
                pass
        return "\n\n".join(parts).strip()
    return str(row.get("question") or src.get("question") or "").strip()


def _prediction_to_faith_row(
    row: Dict[str, Any],
    *,
    mcq_only: Optional[bool] = None,
    strip_prompt: Optional[bool] = None,
    context_mode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    use_mcq = _default_mcq_only() if mcq_only is None else mcq_only
    use_strip = _default_strip_prompt() if strip_prompt is None else strip_prompt
    ctx_mode = _default_faith_context_mode() if context_mode is None else context_mode
    raw_ctx = str(
        row.get("retrieved_context") or row.get("context") or row.get("evidence") or ""
    ).strip()
    if not _rag_flag_true(row):
        norag_ctx = _norag_grounding_context(row)
        if norag_ctx:
            raw_ctx = norag_ctx
    ctx = _prepare_context_for_scoring(raw_ctx, strip_prompt=use_strip)
    mapped: Dict[str, Any] = {
        "id": row.get("question_id", row.get("id", "")),
        "question": row.get("question", ""),
        "context": ctx,
        "context_stored": ctx,
        "model_name": row.get("model_name", ""),
        "rag_flag": row.get("rag_flag"),
        "benchmark": row.get("benchmark", ""),
        "run_folder": row.get("run_folder", ""),
        "source_file": row.get("source_file", ""),
        "question_id": row.get("question_id", ""),
        "_source_row": row,
        "faithfulness_answer_mode": "mcq_option" if use_mcq else "full_response",
        "context_stripped": use_strip,
        "faithfulness_context_mode": ctx_mode,
    }
    mapped.update(_retrieval_fields_from_source(row))
    ans = _build_faithfulness_answer(mapped, mcq_only=use_mcq)
    if not ans:
        return None
    if not ctx:
        return None
    mapped["answer"] = ans
    return mapped


def _default_last_rows() -> int:
    """Keep only the last N raw CSV rows (default 0; set GP_FAITH_LAST_ROWS if needed)."""
    env = os.environ.get("GP_FAITH_LAST_ROWS", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            return 0
    return _KAGGLE_LAST_ROWS


def _resolve_last_rows(last_rows: int) -> int:
    if last_rows >= 0:
        return last_rows
    return _default_last_rows()


def _default_row_offset() -> int:
    """Skip first N RAG work rows after filter (default 0 = full 18180 coverage)."""
    env = os.environ.get("GP_FAITH_ROW_OFFSET", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            return 0
    if os.path.isdir("/kaggle"):
        return _KAGGLE_PREDICTIONS_ROW_OFFSET
    return 0


def _default_start_batch() -> int:
    """1-indexed first batch (default 31 on Kaggle legacy; 11 for sifatali008 account)."""
    env = os.environ.get("GP_FAITH_START_BATCH", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "sifatali008":
        return _SIFATALI_PAPER_START_BATCH
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "fahim220":
        return _FAHIM220_PAPER_START_BATCH
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "fatinshadab":
        return _FATINSHADAB_PAPER_START_BATCH
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "ummesalmahabiba":
        return _UMMESALMAHABIBA_PAPER_START_BATCH
    if os.path.isdir("/kaggle"):
        return _KAGGLE_START_BATCH
    return 1


def _resolve_effective_min_batch_index(
    start_batch_1based: int,
    *,
    batches: List[List[Dict[str, Any]]],
    scored_keys: set,
    only_missing: bool,
) -> int:
    """
    0-based minimum batch index: at least ``start_batch_1based``, or first incomplete batch if later.
    """
    user_min = max(0, int(start_batch_1based) - 1)
    if not only_missing or not scored_keys or not batches:
        return user_min
    first_gap = _find_next_batch_index(
        "",
        len(batches),
        [len(b) for b in batches],
        min_batch_index=0,
        batches=batches,
        scored_keys=scored_keys,
    )
    if first_gap is None:
        return user_min
    return max(user_min, first_gap)


def _resolve_start_batch_index(start_batch: int) -> int:
    """Return 0-based minimum batch index."""
    if start_batch >= 1:
        return max(0, start_batch - 1)
    return max(0, _default_start_batch() - 1)


def _resolve_row_offset(row_offset: int) -> int:
    if row_offset >= 0:
        return row_offset
    return _default_row_offset()


def _slice_last_rows(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return rows
    if n >= len(rows):
        return rows
    return rows[-n:]


def _slice_raw_rows(rows: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    if offset <= 0:
        return rows
    if offset >= len(rows):
        return []
    return rows[offset:]


def load_predictions_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row:
                rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def load_predictions_json(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("rows") or [])


def load_predictions_rows(path: str) -> Tuple[List[Dict[str, Any]], str]:
    if _is_csv_path(path):
        return load_predictions_csv(path), "csv"
    return load_predictions_json(path), "json"


def _default_predictions_path() -> str:
    """Default predictions CSV (Kaggle: fatinshadab or sifatali008 via GP_FAITH_ACCOUNT)."""
    env = os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
    if env:
        return env
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "sifatali008":
        if os.path.isfile(_SIFATALI_PREDICTIONS_CSV):
            return _SIFATALI_PREDICTIONS_CSV
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "fahim220":
        if os.path.isfile(_FAHIM220_PREDICTIONS_CSV):
            return _FAHIM220_PREDICTIONS_CSV
    if os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower() == "ummesalmahabiba":
        if os.path.isfile(_UMMESALMAHABIBA_PREDICTIONS_CSV):
            return _UMMESALMAHABIBA_PREDICTIONS_CSV
    if os.path.isdir("/kaggle"):
        return _KAGGLE_PREDICTIONS_CSV
    local = os.path.join(_script_dir(), "result", "benchmark_results_all_predictions_combined.csv")
    if os.path.isfile(local):
        return local
    return ""


def _resolve_predictions_path(path: str) -> str:
    if path.strip() and os.path.isfile(path.strip()):
        return os.path.abspath(path.strip())
    default = _default_predictions_path()
    if default and os.path.isfile(default):
        print(f"[faithfulness] Using predictions: {default!r}", flush=True)
        return os.path.abspath(default)
    for c in _DEFAULT_PREDICTIONS_CANDIDATES:
        if c and os.path.isfile(c):
            print(f"[faithfulness] Auto-found predictions: {c!r}", flush=True)
            return os.path.abspath(c)
    tried = ", ".join(repr(x) for x in (_KAGGLE_PREDICTIONS_CSV, *_DEFAULT_PREDICTIONS_CANDIDATES[:3]))
    raise FileNotFoundError(
        f"Predictions file not found. Tried: {tried}. "
        "On Kaggle attach Add Data → fatinshadab/benchmark-results-all-predictions-combined "
        "(combined predictions CSV), or set GP_FAITH_PREDICTIONS."
    )


def _default_norag_only() -> bool:
    return os.environ.get("GP_FAITH_NORAG_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def _rows_for_faithfulness(
    raw_rows: List[Dict[str, Any]],
    *,
    rag_only: bool = False,
    norag_only: bool = False,
    mcq_only: Optional[bool] = None,
    strip_prompt: Optional[bool] = None,
    context_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in raw_rows:
        if rag_only and not _rag_flag_true(row):
            continue
        if norag_only and _rag_flag_true(row):
            continue
        mapped = _prediction_to_faith_row(
            row,
            mcq_only=mcq_only,
            strip_prompt=strip_prompt,
            context_mode=context_mode,
        )
        if mapped:
            out.append(mapped)
    return out


def _count_rag_rows_in_predictions(path: str) -> int:
    """Count rag_flag=True rows in the predictions CSV."""
    n = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if _rag_flag_true(row):
                n += 1
    return n


def _count_norag_rows_in_predictions(path: str) -> int:
    """Count rag_flag=False rows in the predictions CSV."""
    n = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if not _rag_flag_true(row):
                n += 1
    return n


def _prepare_faithfulness_work_rows(
    raw_rows: List[Dict[str, Any]],
    *,
    rag_only: bool,
    norag_only: bool = False,
    mcq_only: Optional[bool],
    strip_prompt: Optional[bool],
    context_mode: Optional[str],
    row_offset: int,
    last_rows: int,
) -> List[Dict[str, Any]]:
    """
    Build work rows for scoring. When *rag_only* or *norag_only*, slice offset/limit applies
    to filtered rows (not raw CSV line numbers) so full coverage matches 18,180 per arm.
    """
    work = _rows_for_faithfulness(
        raw_rows,
        rag_only=rag_only,
        norag_only=norag_only,
        mcq_only=mcq_only,
        strip_prompt=strip_prompt,
        context_mode=context_mode,
    )
    if (rag_only or norag_only) and (row_offset > 0 or last_rows > 0):
        if row_offset > 0:
            work = work[row_offset:]
        if last_rows > 0:
            work = work[-last_rows:]
        return work
    if not rag_only and not norag_only:
        sliced = raw_rows
        if last_rows > 0:
            sliced = _slice_last_rows(sliced, last_rows)
        if row_offset > 0:
            sliced = _slice_raw_rows(sliced, row_offset)
        if sliced is not raw_rows:
            return _rows_for_faithfulness(
                sliced,
                rag_only=rag_only,
                norag_only=norag_only,
                mcq_only=mcq_only,
                strip_prompt=strip_prompt,
                context_mode=context_mode,
            )
    return work


def _discover_kaggle_input_faith_dirs() -> List[str]:
    """Find folders under ``/kaggle/input`` that contain faithfulness ``*_batch_*.csv`` files."""
    roots: List[str] = []
    kaggle_in = "/kaggle/input"
    if not os.path.isdir(kaggle_in):
        return roots
    seen: set = set()
    for path in glob.glob(os.path.join(kaggle_in, "**", "*_batch_*.csv"), recursive=True):
        if not _is_faithfulness_batch_csv(path):
            continue
        d = os.path.dirname(os.path.abspath(path))
        if d not in seen:
            seen.add(d)
            roots.append(d)
    for name in ("fathfullness", "Faithfulness", "faithfulness"):
        for path in glob.glob(os.path.join(kaggle_in, "**", name), recursive=True):
            if os.path.isdir(path):
                p = os.path.abspath(path)
                if p not in seen:
                    seen.add(p)
                    roots.append(p)
    return roots


def _apply_default_prior_faith_dirs() -> None:
    """On Kaggle, auto-scan ``/kaggle/input`` for prior faithfulness batch CSV folders."""
    if os.environ.get("GP_FAITH_PRIOR_BATCH_DIRS", "").strip():
        return
    auto: List[str] = []
    if os.path.isdir("/kaggle"):
        auto.extend(_discover_kaggle_input_faith_dirs())
    here = _script_dir()
    for rel in ("fathfullness",):
        p = os.path.join(here, rel)
        if os.path.isdir(p):
            auto.append(p)
    if auto:
        os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = ",".join(dict.fromkeys(auto))
        print(
            f"[faithfulness] Auto prior batch dirs: {os.environ['GP_FAITH_PRIOR_BATCH_DIRS']}",
            flush=True,
        )


def _prior_batch_search_dirs(output_stem: str) -> List[str]:
    """Directories to scan for already-scored batch CSV keys."""
    dirs: List[str] = []
    out_parent = os.path.dirname(os.path.abspath(output_stem)) or "."
    if out_parent:
        dirs.append(out_parent)
    env = os.environ.get("GP_FAITH_PRIOR_BATCH_DIRS", "").strip()
    if env:
        dirs.extend(p.strip() for p in env.split(",") if p.strip())
    here = _script_dir()
    for rel in ("fathfullness", "Faithfulness", "kaggle_working"):
        p = os.path.join(here, rel)
        if os.path.isdir(p) and p not in dirs:
            dirs.append(p)
    if os.path.isdir(_KAGGLE_FAITH_WRITE_DIR):
        for name in os.listdir(_KAGGLE_FAITH_WRITE_DIR):
            p = os.path.join(_KAGGLE_FAITH_WRITE_DIR, name)
            if os.path.isdir(p) and p not in dirs:
                dirs.append(p)
    if os.path.isdir("/kaggle"):
        for p in _discover_kaggle_input_faith_dirs():
            if p not in dirs:
                dirs.append(p)
    return dirs


def _count_faith_batch_files(search_dirs: List[str]) -> Tuple[int, List[str]]:
    """Return (number of batch CSV files, list of search roots used)."""
    seen_dirs: set = set()
    n_files = 0
    used: List[str] = []
    for root in search_dirs:
        root = os.path.abspath(root)
        if root in seen_dirs or not os.path.isdir(root):
            continue
        seen_dirs.add(root)
        hits = [
            p
            for p in glob.glob(os.path.join(root, "**", "*_batch_*.csv"), recursive=True)
            if _is_faithfulness_batch_csv(p)
        ]
        if hits:
            used.append(root)
            n_files += len(hits)
    return n_files, used


def _log_faithfulness_coverage_status(
    *,
    n_scored: int,
    n_total: int,
    target: int,
    n_pending: int,
    n_batches: int,
    batch_size: int,
    next_batch_1based: Optional[int] = None,
    prior_batch_files: int = 0,
    prior_dirs: Optional[List[str]] = None,
    user_start_batch_1based: int = 1,
    prior_match_stats: Optional[Dict[str, int]] = None,
) -> None:
    """Single canonical coverage line (always printed)."""
    batch_part = ""
    if next_batch_1based is not None:
        batch_part = f" | next batch {next_batch_1based}/{n_batches}"

    user_start = max(1, int(user_start_batch_1based))
    assumed_done = 0
    if user_start > 1:
        assumed_done = min((user_start - 1) * batch_size, n_total)
    effective_done = max(n_scored, assumed_done)
    pending_run = max(0, n_total - effective_done)

    if assumed_done > n_scored and n_scored == 0:
        print(
            f"[faithfulness] Coverage: prior_csv=0 keys on disk | "
            f"GP_FAITH_START_BATCH={user_start} → rows 1–{assumed_done} treated as done | "
            f"effective {effective_done}/{n_total} (target {target}) | "
            f"pending_this_run ~{pending_run}{batch_part}",
            flush=True,
        )
    elif assumed_done > n_scored:
        print(
            f"[faithfulness] Coverage: prior_csv={n_scored}/{n_total} keys | "
            f"start_batch={user_start} assumes through row {assumed_done} | "
            f"effective {effective_done}/{n_total} (target {target}) | "
            f"pending {pending_run}{batch_part}",
            flush=True,
        )
    else:
        print(
            f"[faithfulness] Coverage: scored {n_scored}/{n_total} RAG rows "
            f"(target {target}) | pending {n_pending}{batch_part}",
            flush=True,
        )

    if prior_batch_files:
        where = ", ".join(prior_dirs or [])[:500]
        pstats = prior_match_stats or {}
        n_file_keys = pstats.get("prior_keys_in_files", 0)
        n_strict = pstats.get("strict", 0)
        n_relaxed = pstats.get("relaxed", 0)
        print(
            f"[faithfulness] Prior faithfulness batches: {prior_batch_files} file(s), "
            f"{n_file_keys} keys in CSVs → {n_scored} matched to this predictions file "
            f"(strict={n_strict}, relaxed={n_relaxed}) ({where})",
            flush=True,
        )
        if n_file_keys > 0 and n_scored == 0:
            print(
                "[faithfulness] WARNING: prior batch files found but 0 keys match this "
                "predictions CSV (often ``run_folder``/``source_file`` differ across accounts). "
                "Relaxed matching is enabled; if still 0, verify the same combined predictions file.",
                flush=True,
            )
    elif n_scored == 0:
        account = os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower()
        start_env = os.environ.get("GP_FAITH_START_BATCH", "").strip()
        try:
            start_b = max(1, int(start_env)) if start_env else user_start
        except ValueError:
            start_b = user_start
        last_prior_idx = max(0, start_b - 2)
        if start_b > 1:
            print(
                f"[faithfulness] No prior batch CSVs found — skipping batches 1–{start_b - 1} "
                f"via GP_FAITH_START_BATCH only (not key-level resume). "
                f"Recommended: attach batch_0000–batch_{last_prior_idx:04d} and set:\n"
                f"  os.environ['GP_FAITH_PRIOR_BATCH_DIRS'] = '/kaggle/input/YOUR_DATASET'\n"
                f"  # sifatali008: run_faithfulness_sifatali_paper_resume("
                f"start_batch={start_b}, prior_batch_dirs='...')",
                flush=True,
            )
        elif account == "sifatali008":
            print(
                "[faithfulness] No prior batch CSVs found. Fresh sifatali008 run: batch 1/37. "
                "To resume from batch 11, attach batches 0000–0009 and call "
                "run_faithfulness_sifatali_paper_resume(start_batch=11, prior_batch_dirs='...').",
                flush=True,
            )
        elif os.path.isdir("/kaggle"):
            print(
                "[faithfulness] No prior batch CSVs found. Fresh run: batch 1/37. "
                "To resume: attach prior batch CSVs and set GP_FAITH_PRIOR_BATCH_DIRS, or "
                "run_faithfulness_sifatali_paper_resume(...) / GP_FAITH_START_BATCH.",
                flush=True,
            )


def _load_scored_faithfulness_keys(
    output_stem: str = "",
    extra_dirs: Optional[List[str]] = None,
) -> set:
    """Row keys that already have a numeric faithfulness_primary in any batch CSV."""
    return set(_load_scored_faithfulness_records(output_stem, extra_dirs).keys())


def _load_scored_faithfulness_records(
    output_stem: str = "",
    extra_dirs: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Latest scored CSV row per ``_row_key`` from all batch exports."""
    dirs = list(extra_dirs or [])
    if output_stem:
        d = os.path.dirname(os.path.abspath(output_stem)) or "."
        if d not in dirs:
            dirs.append(d)
    dirs.extend(_prior_batch_search_dirs(output_stem or ""))
    seen_dirs: set = set()
    records: Dict[str, Dict[str, Any]] = {}
    for root in dirs:
        root = os.path.abspath(root)
        if root in seen_dirs or not os.path.isdir(root):
            continue
        seen_dirs.add(root)
        pattern = os.path.join(root, "**", "*_batch_*.csv")
        for path in sorted(glob.glob(pattern, recursive=True)):
            if not _is_faithfulness_batch_csv(path):
                continue
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        v = float(row.get("faithfulness_primary", ""))
                        if v != v:
                            continue
                    except (TypeError, ValueError):
                        continue
                    records[_row_key(row)] = row
    return records


def _default_only_missing() -> bool:
    return os.environ.get("GP_FAITH_ONLY_MISSING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_by_configuration(
    results: List[Dict[str, Any]], backend: str, model_id: str = ""
) -> Dict[str, Dict[str, Any]]:
    by_cfg: Dict[str, List[float]] = {}
    for rec in results:
        cfg = str(rec.get("model_name") or rec.get("configuration") or "unknown")
        f = rec.get("faithfulness_primary")
        if isinstance(f, (int, float)) and f == f:
            by_cfg.setdefault(cfg, []).append(float(f))
    out: Dict[str, Dict[str, Any]] = {}
    for cfg, vals in sorted(by_cfg.items()):
        summ = summarize(vals)
        mean = summ["mean"]
        out[cfg] = {
            **summ,
            "faithfulness_mean": mean,
            "hallucination_rate_mean": (100.0 - mean) if mean == mean else None,
            "backend": backend,
            "model": model_id or backend,
        }
    return out


def summarize(vals: List[float]) -> Dict[str, float]:
    clean = [v for v in vals if v == v]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean),
        "n": len(clean),
    }


def _faith_output_dir(stem_name: str) -> str:
    env_out = os.environ.get("GP_FAITH_OUT_DIR", "").strip()
    if env_out:
        env_out = os.path.abspath(env_out)
        # GP_FAITH_OUT_DIR must be a directory, not a predictions CSV path
        if os.path.isfile(env_out) or env_out.lower().endswith(".csv"):
            parent = os.path.dirname(env_out)
            print(
                f"[faithfulness] GP_FAITH_OUT_DIR pointed at a file; using parent dir: {parent!r}",
                flush=True,
            )
            env_out = parent or (
                os.path.join(_KAGGLE_FAITH_WRITE_DIR, stem_name)
                if os.path.isdir("/kaggle")
                else os.path.join(_DEFAULT_FAITH_DIR, stem_name)
            )
        # Kaggle input datasets are read-only — never write batch CSVs there
        if os.path.isdir("/kaggle") and env_out.startswith("/kaggle/input"):
            fallback = os.path.join(_KAGGLE_FAITH_WRITE_DIR, "gap_fill", stem_name)
            print(
                f"[faithfulness] GP_FAITH_OUT_DIR was under /kaggle/input (read-only); "
                f"using {fallback!r}",
                flush=True,
            )
            env_out = fallback
        return env_out
    if os.path.isdir("/kaggle"):
        return os.path.join(_KAGGLE_FAITH_WRITE_DIR, stem_name)
    return os.path.join(_DEFAULT_FAITH_DIR, stem_name)


def _faith_output_stem(
    predictions_path: str,
    out_path: str = "",
    *,
    last_rows: int = 0,
    row_offset: int = 0,
) -> str:
    if out_path.strip():
        base = os.path.splitext(out_path.strip())[0]
        if base.endswith("_faithfulness"):
            return base
        return base + "_faithfulness"
    base = os.path.basename(predictions_path)
    stem_name = f"{os.path.splitext(base)[0]}_faithfulness"
    if last_rows > 0:
        stem_name += f"_last{last_rows}"
    elif row_offset > 0:
        stem_name += f"_from{row_offset}"
    out_dir = _faith_output_dir(stem_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem_name)


def _batch_filename(output_stem: str, batch_index: int) -> str:
    return f"{os.path.basename(output_stem)}_batch_{batch_index:04d}.csv"


def _batch_csv_path(output_stem: str, batch_index: int) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(output_stem)) or ".",
        _batch_filename(output_stem, batch_index),
    )


def _batch_checkpoint_path(output_stem: str, batch_index: int) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(output_stem)) or ".",
        f"{os.path.basename(output_stem)}_batch_{batch_index:04d}.checkpoint.json",
    )


def _count_csv_data_rows(csv_path: str) -> int:
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_valid_faith_rows(csv_path: str) -> int:
    """Rows with a numeric faithfulness_primary (NaN-only batches are not complete)."""
    if not os.path.isfile(csv_path):
        return 0
    n = 0
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            s = row.get("faithfulness_primary", "")
            try:
                v = float(s)
                if v == v:
                    n += 1
            except (TypeError, ValueError):
                continue
    return n


def _batch_is_complete(output_stem: str, batch_index: int, expected_rows: int) -> bool:
    path = _batch_csv_path(output_stem, batch_index)
    if _count_csv_data_rows(path) < expected_rows:
        return False
    return _count_valid_faith_rows(path) >= expected_rows


def _batch_keys_complete(batch_rows: List[Dict[str, Any]], scored_keys: set) -> bool:
    """True when every row in this batch slice already has a score on disk."""
    if not batch_rows:
        return True
    return all(_row_key(r) in scored_keys for r in batch_rows)


def _count_scored_in_work_rows(work_rows: List[Dict[str, Any]], scored_keys: set) -> int:
    return sum(1 for r in work_rows if _row_key(r) in scored_keys)


def _load_faith_batch_records(csv_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row.get("faithfulness_primary", ""))
                if v != v:
                    continue
            except (TypeError, ValueError):
                continue
            out[_row_key(row)] = row
    return out


def _find_next_batch_index(
    output_stem: str,
    n_batches: int,
    batch_sizes: List[int],
    min_batch_index: int = 0,
    *,
    batches: Optional[List[List[Dict[str, Any]]]] = None,
    scored_keys: Optional[set] = None,
) -> Optional[int]:
    start = max(0, min(min_batch_index, n_batches))
    if batches is not None and scored_keys is not None:
        for idx in range(start, n_batches):
            if idx < len(batches) and _batch_keys_complete(batches[idx], scored_keys):
                continue
            return idx
        return None
    for idx in range(start, n_batches):
        if _batch_is_complete(output_stem, idx, batch_sizes[idx]):
            continue
        return idx
    return None


def _delete_faithfulness_batch(output_stem: str, batch_index: int) -> None:
    """Remove one batch CSV + checkpoint so it can be re-scored."""
    for path in (
        _batch_csv_path(output_stem, batch_index),
        _batch_checkpoint_path(output_stem, batch_index),
    ):
        if os.path.isfile(path):
            os.remove(path)
            print(f"[faithfulness] removed {path!r}", flush=True)


def _resolve_force_batch_index(
    n_batches: int, force_batch_index: Optional[int] = None
) -> Optional[int]:
    raw = os.environ.get("GP_FAITH_FORCE_BATCH", "").strip()
    idx = force_batch_index
    if idx is None and raw.lstrip("-").isdigit():
        idx = int(raw)
    if idx is None:
        return None
    if idx < 0 or idx >= n_batches:
        raise ValueError(f"force batch index {idx} out of range 0..{n_batches - 1}")
    return idx


def _split_batches(rows: List[Dict[str, Any]], batch_size: int) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    batches: List[List[Dict[str, Any]]] = []
    sizes: List[int] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batches.append(chunk)
        sizes.append(len(chunk))
    return batches, sizes


def _work_fingerprint(rows: List[Dict[str, Any]]) -> str:
    keys = sorted(_row_key(r) for r in rows)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]


def _run_config_meta(
    *,
    predictions_path: str,
    source_format: str,
    backend: str,
    model_id: str,
    rag_only: bool,
    batch_size: int,
    batch_index: Optional[int],
    output_stem: str,
    work_rows: List[Dict[str, Any]],
    n_input_rows: int,
    max_ctx_chars: int,
    mcq_only: bool,
    strip_prompt: bool,
    faithfulness_context_mode: str,
) -> Dict[str, Any]:
    return {
        "predictions_path": predictions_path,
        "source_format": source_format,
        "backend": backend,
        "model_id": model_id,
        "rag_only": rag_only,
        "batch_size": batch_size,
        "batch_index": batch_index,
        "output_stem": output_stem,
        "work_fingerprint": _work_fingerprint(work_rows),
        "n_work_rows": len(work_rows),
        "n_input_rows": n_input_rows,
        "max_ctx_chars": max_ctx_chars,
        "mcq_only": mcq_only,
        "strip_prompt": strip_prompt,
        "faithfulness_context_mode": faithfulness_context_mode,
    }


def _checkpoint_compatible(ckpt_meta: Dict[str, Any], run_meta: Dict[str, Any]) -> bool:
    keys = (
        "predictions_path",
        "source_format",
        "backend",
        "model_id",
        "rag_only",
        "batch_size",
        "output_stem",
        "batch_index",
        "work_fingerprint",
        "max_ctx_chars",
        "mcq_only",
        "strip_prompt",
        "faithfulness_context_mode",
    )
    for k in keys:
        if ckpt_meta.get(k) != run_meta.get(k):
            return False
    return True


def _load_batch_checkpoint(
    ckpt_path: str, run_meta: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    if not os.path.isfile(ckpt_path):
        return {}, 0
    with open(ckpt_path, encoding="utf-8") as f:
        payload = json.load(f)
    ckpt_meta = payload.get("meta") or {}
    if not _checkpoint_compatible(ckpt_meta, run_meta):
        print(
            "[faithfulness] Checkpoint options differ from this run; starting fresh.",
            flush=True,
        )
        return {}, 0
    by_key: Dict[str, Dict[str, Any]] = {}
    errors = 0
    for r in payload.get("rows") or []:
        if not isinstance(r, dict):
            continue
        s = r.get("faithfulness_primary", "")
        try:
            v = float(s)
            if v != v:
                continue
        except (TypeError, ValueError):
            continue
            by_key[_row_key(r)] = r
    print(
        f"[faithfulness] Resuming from {ckpt_path!r}: "
        f"{len(by_key)}/{run_meta.get('n_work_rows', '?')} rows already scored.",
        flush=True,
    )
    return by_key, errors


def _save_batch_checkpoint(
    ckpt_path: str,
    *,
    run_meta: Dict[str, Any],
    by_key: Dict[str, Dict[str, Any]],
    work_rows: List[Dict[str, Any]],
    parse_failures: int,
) -> None:
    ordered = [by_key[_row_key(r)] for r in work_rows if _row_key(r) in by_key]
    payload = {
        "meta": {
            **run_meta,
            "n_scored": len(ordered),
            "parse_failures": parse_failures,
            "checkpoint": True,
        },
        "rows": ordered,
    }
    od = os.path.dirname(os.path.abspath(ckpt_path))
    if od:
        os.makedirs(od, exist_ok=True)
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ckpt_path)


def _result_record(
    row: Dict[str, Any],
    score: float,
    note: str,
    model_id: str,
    *,
    context_source: str = "",
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "id": row.get("id", ""),
        "run_folder": row.get("run_folder", ""),
        "source_file": row.get("source_file", ""),
        "benchmark": row.get("benchmark", ""),
        "question_id": row.get("question_id", row.get("id", "")),
        "model_name": row.get("model_name", ""),
        "rag_flag": row.get("rag_flag"),
        "faithfulness_primary": score,
        "note_primary": note,
        "hallucination_rate_primary": (100.0 - score) if score == score else None,
        "faithfulness_model": model_id,
        "faithfulness_answer_mode": row.get("faithfulness_answer_mode", ""),
        "context_stripped": row.get("context_stripped", ""),
        "faithfulness_context_mode": row.get("faithfulness_context_mode", ""),
        "faithfulness_context_source": context_source or row.get("faithfulness_context_source", ""),
    }
    for k in (
        "evidence_token_overlap",
        "gold_chunk_found",
        "mcq_correct",
        "label_correct",
        "parsed_prediction",
        "recall_at_1",
        "recall_at_5",
        "rag_context_used",
    ):
        if k in row and row.get(k) not in ("", None):
            rec[k] = row.get(k)
    try:
        ov = float(row.get("evidence_token_overlap"))
        if ov == ov:
            rec["faithfulness_overlap_pct"] = round(max(0.0, min(100.0, ov * 100.0)), 2)
    except (TypeError, ValueError):
        pass
    return rec


def _print_batch_grounding_summary(out_rows: List[Dict[str, Any]]) -> None:
    if not out_rows:
        return
    scores: List[float] = []
    overlaps: List[float] = []
    gold_yes = 0
    gold_n = 0
    for r in out_rows:
        f = r.get("faithfulness_primary")
        if isinstance(f, (int, float)) and f == f:
            scores.append(float(f))
        o = r.get("evidence_token_overlap")
        try:
            ov = float(o)
            if ov == ov:
                overlaps.append(ov)
        except (TypeError, ValueError):
            pass
        g = r.get("gold_chunk_found")
        if g in (True, "True", "true", "1", 1):
            gold_yes += 1
            gold_n += 1
        elif g not in ("", None):
            gold_n += 1
    parts = []
    if scores:
        parts.append(f"mean_faithfulness={statistics.mean(scores):.1f}")
        parts.append(f"pct_score_zero={100.0 * sum(1 for s in scores if s <= 0) / len(scores):.1f}%")
    if overlaps:
        parts.append(f"mean_evidence_token_overlap={statistics.mean(overlaps):.4f}")
    if gold_n:
        parts.append(f"pct_gold_chunk_found={100.0 * gold_yes / gold_n:.1f}%")
    ctx_counts = Counter(
        str(r.get("faithfulness_context_source") or "") for r in out_rows
    )
    if ctx_counts:
        top_ctx = ", ".join(f"{k}={v}" for k, v in ctx_counts.most_common(4) if k)
        if top_ctx:
            parts.append(f"context_sources({top_ctx})")
    by_bench: Dict[str, List[float]] = {}
    for r in out_rows:
        b = str(r.get("benchmark") or "")
        f = r.get("faithfulness_primary")
        if b and isinstance(f, (int, float)) and f == f:
            by_bench.setdefault(b, []).append(float(f))
    if by_bench:
        bench_means = ", ".join(
            f"{b}={statistics.mean(v):.1f}" for b, v in sorted(by_bench.items())
        )
        parts.append(f"by_benchmark({bench_means})")
    ov_pcts: List[float] = []
    for r in out_rows:
        v = r.get("faithfulness_overlap_pct")
        if v is None:
            try:
                ov = float(r.get("evidence_token_overlap"))
                if ov == ov:
                    v = ov * 100.0
            except (TypeError, ValueError):
                continue
        try:
            fv = float(v)
            if fv == fv:
                ov_pcts.append(fv)
        except (TypeError, ValueError):
            pass
    if ov_pcts:
        parts.append(f"mean_overlap_pct={statistics.mean(ov_pcts):.2f}")
    if parts:
        print(f"[faithfulness] Batch summary: {'; '.join(parts)}", flush=True)
        print(
            "[faithfulness] Note: low judge scores often mean weak retrieval, not only "
            "wrong MCQ letter. Compare mean_faithfulness vs mean_overlap_pct.",
            flush=True,
        )


def _write_faith_batch_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen and not k.startswith("_"):
                seen.add(k)
                fieldnames.append(k)
    od = os.path.dirname(os.path.abspath(csv_path))
    if od:
        os.makedirs(od, exist_ok=True)
    tmp = csv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, csv_path)


def _execute_faithfulness_loop(
    *,
    work_rows: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
    ckpt_path: str,
    backend: str,
    model_id: str,
    api_key: str,
    openrouter_model: str,
    max_ctx_chars: int,
    resume: bool,
    checkpoint_every: int,
    deepeval_metric=None,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    by_key: Dict[str, Dict[str, Any]] = {}
    errors = 0
    resumed = False
    if resume:
        by_key, errors = _load_batch_checkpoint(ckpt_path, run_meta)
        resumed = len(by_key) > 0

    pending = [r for r in work_rows if _row_key(r) not in by_key]
    n_total = len(work_rows)
    n_done = len(by_key)

    mcq_only = bool(run_meta.get("mcq_only", _default_mcq_only()))
    strip_prompt = bool(run_meta.get("strip_prompt", _default_strip_prompt()))
    ctx_mode = str(
        run_meta.get("faithfulness_context_mode") or _default_faith_context_mode()
    )
    # Load embedder / FAISS before the judge model to reduce T4 OOM (or skip if no indexes).
    _init_faithfulness_rag(ctx_mode)
    if backend == "local_hf":
        _try_load_hf_token()
        _get_local_hf_model(model_id)
    elif backend == "deepeval" and deepeval_metric is None:
        _try_load_hf_token()
        deepeval_metric = _get_deepeval_metric(model_id)

    for i, row in enumerate(pending):
        ctx, ctx_src = _resolve_scoring_context(
            row, ctx_mode, strip_prompt=strip_prompt
        )
        ans = str(row.get("answer", "")).strip()
        q = str(row.get("question", "")).strip()
        if not ctx or not ans:
            errors += 1
            rec = _result_record(row, float("nan"), "empty_context_or_answer", model_id, context_source=ctx_src)
            by_key[_row_key(row)] = rec
            continue
        score, note = score_one(
            api_key,
            openrouter_model,
            q,
            ctx,
            ans,
            backend=backend,
            deepeval_metric=deepeval_metric,
            model_id=model_id,
            max_ctx_chars=max_ctx_chars,
            mcq_only=mcq_only,
        )
        if score != score:
            errors += 1
        rec = _result_record(row, score, note, model_id, context_source=ctx_src)
        by_key[_row_key(row)] = rec
        global_idx = n_done + i + 1
        if score == score:
            ctx_hint = ""
            if global_idx <= 5 or os.environ.get("GP_FAITH_LOG_CTX", "").strip() in (
                "1",
                "true",
                "yes",
            ):
                ctx_hint = f" ctx={ctx_src}"
            print(
                f"  [{global_idx}/{n_total}] {row.get('model_name')} "
                f"faithfulness={score:.1f}{ctx_hint}",
                flush=True,
            )
        else:
            print(
                f"  [{global_idx}/{n_total}] {row.get('model_name')} faithfulness=nan "
                f"({(note or 'no note')[:120]})",
                flush=True,
            )
        if (i + 1) % max(1, checkpoint_every) == 0 or i == len(pending) - 1:
            _save_batch_checkpoint(
                ckpt_path,
                run_meta=run_meta,
                by_key=by_key,
                work_rows=work_rows,
                parse_failures=errors,
            )

    out_rows = [by_key[_row_key(r)] for r in work_rows if _row_key(r) in by_key]
    return out_rows, errors, resumed


def run_faithfulness_batch(
    predictions_path: str,
    *,
    batch_size: int = 500,
    rag_only: bool = True,
    norag_only: bool = False,
    backend: str = "",
    model_id: str = "",
    openrouter_model: str = "",
    max_ctx_chars: int = 6000,
    row_limit: int = 0,
    row_offset: int = -1,
    last_rows: int = -1,
    start_batch: int = -1,
    resume: bool = True,
    checkpoint_every: int = 1,
    mcq_only: Optional[bool] = None,
    strip_prompt: Optional[bool] = None,
    context_mode: Optional[str] = None,
    force_batch_index: Optional[int] = None,
) -> Dict[str, Any]:
    _apply_default_prior_faith_dirs()
    predictions_path = _resolve_predictions_path(predictions_path)
    raw_rows, source_format = load_predictions_rows(predictions_path)
    n_raw_total = len(raw_rows)
    resolved_last = _resolve_last_rows(last_rows)
    resolved_offset = _resolve_row_offset(row_offset)
    use_mcq = _default_mcq_only() if mcq_only is None else mcq_only
    use_strip = _default_strip_prompt() if strip_prompt is None else strip_prompt
    use_ctx = (
        os.environ.get("GP_FAITH_CONTEXT", "").strip().lower()
        or (_default_faith_context_mode() if context_mode is None else context_mode)
    )
    norag_only = norag_only or _default_norag_only()
    if rag_only and norag_only:
        raise ValueError("rag_only and norag_only are mutually exclusive")
    if norag_only:
        use_ctx = "stored"
    _warn_faithfulness_context_setup(use_ctx)
    work_rows = _prepare_faithfulness_work_rows(
        raw_rows,
        rag_only=rag_only,
        norag_only=norag_only,
        mcq_only=use_mcq,
        strip_prompt=use_strip,
        context_mode=use_ctx,
        row_offset=resolved_offset,
        last_rows=resolved_last,
    )
    n_work_eligible = len(work_rows)
    if rag_only:
        target_rows = _count_rag_rows_in_predictions(predictions_path)
        arm_label = "RAG"
    elif norag_only:
        target_rows = _count_norag_rows_in_predictions(predictions_path)
        arm_label = "NoRAG"
    else:
        target_rows = n_work_eligible
        arm_label = "all"
    if row_limit and row_limit > 0:
        work_rows = work_rows[:row_limit]
    output_stem = _faith_output_stem(
        predictions_path, last_rows=resolved_last, row_offset=resolved_offset
    )
    work_rows_full = work_rows
    n_total_work = len(work_rows_full)
    only_missing = _default_only_missing()
    scored_keys: set = set()
    scored_records: Dict[str, Dict[str, Any]] = {}
    prior_dirs = _prior_batch_search_dirs(output_stem)
    n_prior_files, prior_used = _count_faith_batch_files(prior_dirs)
    if only_missing:
        scored_records = _load_scored_faithfulness_records(output_stem)
        scored_keys, prior_match_stats = _work_row_keys_covered_by_prior(
            work_rows_full, scored_records
        )
    else:
        prior_match_stats = {}
    n_scored = len(scored_keys) if scored_keys else 0
    n_pending_total = n_total_work - n_scored
    work_rows = work_rows_full
    n_input = len(work_rows)
    if n_input == 0:
        if only_missing and scored_keys and (rag_only or norag_only) and n_scored >= target_rows:
            print(
                f"[faithfulness] All {target_rows} {arm_label} rows already scored "
                f"({n_scored}/{n_total_work} keys on disk). Run build_deduped_faithfulness_final().",
                flush=True,
            )
            return {
                "meta": {
                    "complete": True,
                    "n_rows": 0,
                    "n_scored": n_scored,
                    "n_total_work": n_total_work,
                }
            }
        raise ValueError(
            "No faithfulness-eligible rows (need context + answer; check rag_only/norag_only)."
        )

    mid = _resolve_faithfulness_model(model_id)
    resolved_backend, coerced_model = _normalize_faithfulness_run(backend, model_id)
    if coerced_model:
        mid = _resolve_faithfulness_model(coerced_model)
    resolved_backend = resolved_backend or "local_hf"
    if resolved_backend == "auto":
        resolved_backend = "local_hf"

    _log_backend(resolved_backend, mid)
    batches, batch_sizes = _split_batches(work_rows, batch_size)
    n_batches = len(batches)
    user_start_batch_1 = _user_start_batch_1based(start_batch)
    print(
        f"[faithfulness] Start batch resolved: {user_start_batch_1}/{n_batches} "
        f"(GP_FAITH_START_BATCH={os.environ.get('GP_FAITH_START_BATCH', '')!r}, "
        f"arg={start_batch if start_batch >= 1 else '—'})",
        flush=True,
    )
    min_batch_idx = _resolve_effective_min_batch_index(
        user_start_batch_1,
        batches=batches,
        scored_keys=scored_keys,
        only_missing=only_missing,
    )
    start_batch_1 = min_batch_idx + 1
    if min_batch_idx > 0:
        row_end_skip = min(min_batch_idx * batch_size, n_total_work)
        row_start_run = row_end_skip + 1 if row_end_skip < n_total_work else n_total_work
        print(
            f"[faithfulness] Starting at batch {start_batch_1}/{n_batches} "
            f"(skip batches 1-{min_batch_idx}, rows 1-{row_end_skip} assumed done; "
            f"this run rows {row_start_run}-{n_total_work}, "
            f"{n_batches - min_batch_idx} batches)",
            flush=True,
        )
    forced = _resolve_force_batch_index(n_batches, force_batch_index)
    if forced is not None:
        next_idx = forced
        print(
            f"[faithfulness] Forced batch {next_idx} only (rerun; batches 1+ unchanged on disk).",
            flush=True,
        )
    else:
        next_idx = _find_next_batch_index(
            output_stem,
            n_batches,
            batch_sizes,
            min_batch_index=min_batch_idx,
            batches=batches,
            scored_keys=scored_keys if (only_missing and scored_keys) else None,
        )
    floor_idx = max(0, user_start_batch_1 - 1)
    if next_idx is not None and next_idx < floor_idx:
        print(
            f"[faithfulness] Clamping batch {next_idx + 1} → {floor_idx + 1} "
            f"(GP_FAITH_START_BATCH={user_start_batch_1}; ignore stale batch_0000–"
            f"{floor_idx - 1:04d} from a fresh-run mistake)",
            flush=True,
        )
        next_idx = floor_idx
    work_dir = os.path.dirname(os.path.abspath(output_stem))
    next_batch_1 = (next_idx + 1) if next_idx is not None else None
    _log_faithfulness_coverage_status(
        n_scored=n_scored,
        n_total=n_total_work,
        target=target_rows,
        n_pending=n_pending_total,
        n_batches=n_batches,
        batch_size=batch_size,
        next_batch_1based=next_batch_1,
        prior_batch_files=n_prior_files,
        prior_dirs=prior_used,
        user_start_batch_1based=user_start_batch_1,
        prior_match_stats=prior_match_stats,
    )

    if only_missing and scored_keys is not None:
        completed = n_scored
    else:
        completed = sum(
            batch_sizes[i]
            for i in range(n_batches)
            if _batch_is_complete(output_stem, i, batch_sizes[i])
        )
    print(
        f"[faithfulness] Loaded {n_raw_total} rows ({source_format}) from {predictions_path!r}"
        + (f"; {arm_label} work rows={n_work_eligible}" if (rag_only or norag_only) else "")
        + f"; {n_batches} batches @ {batch_size}; "
        f"output_dir={work_dir!r}; "
        f"mcq_only={use_mcq} context_mode={use_ctx}; "
        f"model={mid!r}.",
        flush=True,
    )
    if completed > 0 and next_idx is not None:
        print(
            "[faithfulness] Tip: delete existing batch CSV/checkpoints if you changed "
            "mcq_only/strip_prompt/context_mode (checkpoint will not resume across rubric changes).",
        flush=True,
    )

    if next_idx is None:
        manifest_path = f"{output_stem}_batch_manifest.json"
        manifest = {
            "predictions_path": predictions_path,
            "output_stem": output_stem,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "n_rows": n_total_work,
            "n_scored": n_scored,
            "complete": True,
            "batch_files": [_batch_csv_path(output_stem, i) for i in range(n_batches)],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(
            f"[faithfulness] All {n_batches} batches complete "
            f"({n_scored}/{n_total_work} {arm_label} rows scored). Manifest: {manifest_path}",
            flush=True,
        )
        return {"meta": manifest, "complete": True}

    batch_rows = batches[next_idx]
    pending_rows = (
        [r for r in batch_rows if _row_key(r) not in scored_keys]
        if only_missing and scored_keys
        else batch_rows
    )
    batch_csv = _batch_csv_path(output_stem, next_idx)
    ckpt_path = _batch_checkpoint_path(output_stem, next_idx)
    run_meta = _run_config_meta(
        predictions_path=predictions_path,
        source_format=source_format,
        backend=resolved_backend,
        model_id=mid,
        rag_only=rag_only,
        batch_size=batch_size,
        batch_index=next_idx,
        output_stem=output_stem,
        work_rows=pending_rows,
        n_input_rows=n_total_work,
        max_ctx_chars=max_ctx_chars,
        mcq_only=use_mcq,
        strip_prompt=use_strip,
        faithfulness_context_mode=use_ctx,
    )
    run_meta["norag_only"] = norag_only

    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    or_model = (openrouter_model or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")).strip()
    deepeval_metric = _get_deepeval_metric(mid) if resolved_backend == "deepeval" else None

    print(
        f"[faithfulness] Processing batch {next_idx + 1}/{n_batches} "
        f"({len(batch_rows)} rows in slice; {len(pending_rows)} to score; "
        f"{max(n_scored, min((user_start_batch_1 - 1) * batch_size, n_total_work) if user_start_batch_1 > 1 else 0)}/{n_total_work} effective done, "
        f"{n_scored} from prior CSVs) -> {batch_csv!r}",
        flush=True,
    )

    out_rows, errors, resumed = _execute_faithfulness_loop(
        work_rows=pending_rows,
        run_meta=run_meta,
        ckpt_path=ckpt_path,
        backend=resolved_backend,
        model_id=mid,
        api_key=key,
        openrouter_model=or_model,
        max_ctx_chars=max_ctx_chars,
        resume=resume,
        checkpoint_every=checkpoint_every,
        deepeval_metric=deepeval_metric,
    )

    by_key_out = _load_faith_batch_records(batch_csv)
    for rec in out_rows:
        by_key_out[_row_key(rec)] = rec
    merged_rows: List[Dict[str, Any]] = []
    for r in batch_rows:
        k = _row_key(r)
        if k in by_key_out:
            merged_rows.append(by_key_out[k])
        elif _prior_record_for_work_row(r, scored_records) is not None:
            merged_rows.append(_prior_record_for_work_row(r, scored_records))

    _write_faith_batch_csv(batch_csv, merged_rows)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)

    _print_batch_grounding_summary(out_rows)
    keys_after = scored_keys | {_row_key(rec) for rec in merged_rows}
    n_scored_after = _count_scored_in_work_rows(work_rows_full, keys_after)
    remaining_batches = sum(
        1 for i in range(n_batches) if not _batch_keys_complete(batches[i], keys_after)
    )
    print(
        f"[faithfulness] Wrote {batch_csv!r} ({len(merged_rows)} rows in file, "
        f"{len(pending_rows)} newly scored, failures={errors}). "
        f"Progress: {n_scored_after}/{n_total_work} {arm_label} rows; "
        f"{remaining_batches} batch(es) still incomplete. Re-run for next batch.",
        flush=True,
    )

    return {
        "meta": {
            **run_meta,
            "batch_csv": batch_csv,
            "n_scored": len(out_rows),
            "n_scored_total": n_scored_after,
            "n_total_work": n_total_work,
            "arm_label": arm_label,
            "failures": errors,
            "resumed": resumed,
            "n_batches_total": n_batches,
            "n_batches_remaining": remaining_batches,
            "n_rows_done_total": n_scored_after,
        },
        "rows": merged_rows,
    }


def _load_openrouter_keys_from_notebook() -> None:
    primary_labels = ("OPENROUTER_API_KEY", "OPENROUTER_GAMMA", "OPENROUTER_LLAMA")
    alt_labels = ("OPENROUTER_API_KEY_ALT", "OPENROUTER_LLAMA", "OPENROUTER_GAMMA")

    def _try_kaggle() -> None:
        try:
            from kaggle_secrets import UserSecretsClient

            c = UserSecretsClient()
            if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
                for name in primary_labels:
                    try:
                        v = c.get_secret(name)
                        if v and str(v).strip():
                            os.environ["OPENROUTER_API_KEY"] = str(v).strip()
                            print(f"Loaded OPENROUTER_API_KEY from Kaggle secret: {name}", flush=True)
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    _try_kaggle()


def run_faithfulness_eval(
    rows: List[Dict[str, Any]],
    *,
    out_path: str = "",
    backend: str = "auto",
    model: str = "",
    model_alt: str = "",
    api_key: str = "",
    api_key_alt: str = "",
    sleep_s: float = 0.5,
    max_ctx_chars: int = 6000,
    mcq_only: Optional[bool] = None,
    context_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Score rows and return full payload (single-shot, no batching)."""
    use_mcq = _default_mcq_only() if mcq_only is None else mcq_only
    use_strip = _default_strip_prompt()
    ctx_mode = _default_faith_context_mode() if context_mode is None else context_mode
    _init_faithfulness_rag(ctx_mode)
    if rows:
        mode = str(rows[0].get("faithfulness_answer_mode") or "")
        if mode == "full_response":
            use_mcq = False
        elif mode == "mcq_option":
            use_mcq = True
    resolved_backend, coerced_model = _normalize_faithfulness_run(backend or "auto", model)
    if resolved_backend == "auto":
        resolved_backend = "local_hf"

    mid = _resolve_faithfulness_model(coerced_model)
    key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    key_alt = (api_key_alt or os.environ.get("OPENROUTER_API_KEY_ALT") or "").strip()
    judge_model = (model or os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")).strip()
    judge_model_alt = (model_alt or os.environ.get("OPENROUTER_MODEL_ALT", "") or judge_model).strip()

    if resolved_backend == "openrouter" and not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY required for openrouter backend. "
            "Use --backend deepeval or set OPENROUTER_API_KEY."
        )

    _log_backend(resolved_backend, mid if resolved_backend in ("deepeval", "local_hf") else judge_model)
    deepeval_metric = _get_deepeval_metric(mid) if resolved_backend == "deepeval" else None
    if resolved_backend in ("deepeval", "local_hf"):
        _try_load_hf_token()
    if resolved_backend == "local_hf":
        _get_local_hf_model(mid)
    elif resolved_backend == "deepeval":
        pass

    results: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    alt_scores: List[float] = []

    for i, row in enumerate(rows):
        ctx, ctx_src = _resolve_scoring_context(row, ctx_mode, strip_prompt=use_strip)
        ans = str(row.get("answer", "")).strip()
        q = str(row.get("question", "")).strip()
        rid = row.get("id", i)
        if not ctx or not ans:
            continue

        f1, n1 = score_one(
            key,
            judge_model,
            q,
            ctx,
            ans,
            backend=resolved_backend,
            deepeval_metric=deepeval_metric,
            model_id=mid,
            max_ctx_chars=max_ctx_chars,
            mcq_only=use_mcq,
        )
        primary_scores.append(f1)
        rec = _result_record(
            row,
            f1,
            n1,
            mid if resolved_backend in ("deepeval", "local_hf") else judge_model,
            context_source=ctx_src,
        )
        if key_alt and resolved_backend == "openrouter":
            f2, n2 = score_one_openrouter(
                key_alt,
                judge_model_alt,
                q,
                ctx,
                ans,
                max_ctx_chars=max_ctx_chars,
                mcq_only=use_mcq,
            )
            alt_scores.append(f2)
            rec["faithfulness_alt"] = f2
            rec["note_alt"] = n2
            rec["hallucination_rate_alt"] = (100.0 - f2) if f2 == f2 else None

        results.append(rec)
        disp = f"{f1:.1f}" if f1 == f1 else "nan"
        print(f"[{i+1}/{len(rows)}] id={rid} cfg={rec.get('model_name')} faithfulness={disp}", flush=True)
        if sleep_s > 0 and resolved_backend == "openrouter":
            time.sleep(sleep_s)

    model_label = mid if resolved_backend == "deepeval" else judge_model
    out: Dict[str, Any] = {
        "meta": {
            "backend": resolved_backend,
            "model_primary": model_label,
            "model_alt": judge_model_alt if key_alt else None,
            "n_rows": len(results),
        },
        "summary_primary": summarize(primary_scores),
        "summary_alt": summarize(alt_scores) if key_alt else None,
        "aggregate_by_configuration": aggregate_by_configuration(results, resolved_backend, model_label),
        "rows": results,
    }
    if out_path:
        od = os.path.dirname(os.path.abspath(out_path))
        if od:
            os.makedirs(od, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote faithfulness results → {out_path}", flush=True)
    return out


def _register_pasted_notebook_module() -> None:
    """
    Kaggle/Jupyter: pasting this file runs as ``__main__``, not as a module.

    Register ``__main__`` as ``run_faithfulness_eval`` so
    ``from run_faithfulness_eval import ...`` works in later cells.
    """
    if __name__ != "__main__":
        return
    import sys

    main = sys.modules.get("__main__")
    if main is not None:
        sys.modules["run_faithfulness_eval"] = main


def _in_jupyter_or_ipython() -> bool:
    if any(m in sys.modules for m in ("IPython", "ipykernel")):
        return True
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def _pip_install_packages(packages: List[str]) -> None:
    if not packages:
        return
    import subprocess

    print(f"[faithfulness] Installing: {', '.join(packages)}", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
    )


def _ensure_faithfulness_deps(backend: str = "") -> None:
    """Install deps when pasted in a notebook without a prior pip cell."""
    if os.environ.get("GP_FAITH_NO_AUTO_PIP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    b = (backend or _default_backend()).strip().lower()
    pkgs = ["transformers", "accelerate"]
    if b == "deepeval":
        pkgs = ["deepeval", *pkgs]
    missing: List[str] = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    _pip_install_packages(missing)


def _ensure_rag_retrieval_deps() -> bool:
    """Install faiss-cpu + sentence-transformers for built-in FAISS re-retrieval."""
    if _inline_rag_deps_ok():
        return True
    if os.environ.get("GP_FAITH_NO_AUTO_PIP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    missing: List[str] = []
    try:
        import faiss  # noqa: F401
    except ImportError:
        missing.append("faiss-cpu")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")
    if not missing:
        return True
    try:
        _pip_install_packages(missing)
    except Exception as exc:
        print(f"[faithfulness] RAG pip install failed: {exc}", flush=True)
        return False
    ok = _inline_rag_deps_ok()
    if ok:
        print("[faithfulness] RAG retrieval deps ready (faiss-cpu, sentence-transformers).", flush=True)
    return ok


def rerun_faithfulness_batch(
    batch_index: int = 0,
    predictions_path: str = "",
    *,
    batch_size: int = -1,
    rag_only: bool = True,
    backend: str = "",
    model: str = "",
    resume: bool = False,
    checkpoint_every: int = 1,
    mcq_only: Optional[bool] = None,
    strip_prompt: Optional[bool] = None,
    context_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete and re-score a single batch (e.g. batch 0 after PubMedQA gold-index fix).

    Does not touch other batch CSVs on disk.
    """
    _try_load_hf_token()
    path = _resolve_predictions_path(predictions_path or _default_predictions_path())
    output_stem = _faith_output_stem(path)
    _delete_faithfulness_batch(output_stem, batch_index)
    bs = batch_size if batch_size >= 0 else _default_batch_size()
    b, mid_in = _normalize_faithfulness_run(
        (backend or _default_backend()).strip(),
        model.strip(),
    )
    mid = _resolve_faithfulness_model(mid_in)
    use_mcq = _default_mcq_only() if mcq_only is None else mcq_only
    use_strip = _default_strip_prompt() if strip_prompt is None else strip_prompt
    use_ctx = (
        os.environ.get("GP_FAITH_CONTEXT", "").strip().lower()
        or (_default_faith_context_mode() if context_mode is None else context_mode)
    )
    return run_faithfulness_batch(
        path,
        batch_size=bs,
        rag_only=rag_only,
        backend=b,
        model_id=mid,
        max_ctx_chars=_default_max_ctx_chars(),
        resume=resume,
        checkpoint_every=max(1, checkpoint_every),
        mcq_only=use_mcq,
        strip_prompt=use_strip,
        context_mode=use_ctx,
        force_batch_index=batch_index,
    )


def run_faithfulness(
    predictions_path: str = "",
    *,
    batch_size: int = -1,
    rag_only: bool = True,
    norag_only: bool = False,
    backend: str = "",
    model: str = "",
    limit: int = 0,
    row_offset: int = -1,
    last_rows: int = -1,
    start_batch: int = -1,
    max_ctx_chars: int = -1,
    resume: bool = True,
    checkpoint_every: int = 1,
    mcq_only: Optional[bool] = None,
    strip_prompt: Optional[bool] = None,
    context_mode: Optional[str] = None,
    force_batch_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Notebook-friendly entry point (mirrors ``judge_predictions`` in llm_judge.py).

    Default: one batch of 500 RAG rows on Kaggle GPU with local_hf + Qwen2.5-3B.
    """
    _try_load_hf_token()
    bs = batch_size if batch_size >= 0 else _default_batch_size()
    b, mid_in = _normalize_faithfulness_run(
        (backend or _default_backend()).strip(),
        model.strip(),
    )
    mid = _resolve_faithfulness_model(mid_in)
    mc = max_ctx_chars if max_ctx_chars >= 0 else _default_max_ctx_chars()
    use_mcq = _default_mcq_only() if mcq_only is None else mcq_only
    use_strip = _default_strip_prompt() if strip_prompt is None else strip_prompt
    use_ctx = (
        os.environ.get("GP_FAITH_CONTEXT", "").strip().lower()
        or (_default_faith_context_mode() if context_mode is None else context_mode)
    )
    _warn_faithfulness_context_setup(use_ctx)
    path = _resolve_predictions_path(predictions_path or _default_predictions_path())
    if bs > 0:
        return run_faithfulness_batch(
            path,
            batch_size=bs,
            rag_only=rag_only,
            norag_only=norag_only,
            backend=b,
            model_id=mid,
            max_ctx_chars=mc,
            row_limit=limit,
            row_offset=row_offset,
            last_rows=last_rows,
            start_batch=start_batch,
            resume=resume,
            checkpoint_every=max(1, checkpoint_every),
            mcq_only=use_mcq,
            strip_prompt=use_strip,
            context_mode=use_ctx,
            force_batch_index=force_batch_index,
        )
    raw_rows, _ = load_predictions_rows(path)
    resolved_last = _resolve_last_rows(last_rows)
    resolved_offset = _resolve_row_offset(row_offset)
    rows = _prepare_faithfulness_work_rows(
        raw_rows,
        rag_only=rag_only,
        norag_only=norag_only,
        mcq_only=use_mcq,
        strip_prompt=use_strip,
        context_mode=use_ctx,
        row_offset=resolved_offset,
        last_rows=resolved_last,
    )
    if limit > 0:
        rows = rows[:limit]
    out = run_faithfulness_eval(
        rows,
        backend=b,
        model=mid,
        max_ctx_chars=mc,
        mcq_only=use_mcq,
    )
    if out.get("rows"):
        _print_batch_grounding_summary(out["rows"])
    return out


def _maybe_notebook_autorun() -> None:
    """
    Run one faithfulness batch after paste-in-cell load (default in Jupyter / IPython).

    ``GP_FAITH_AUTO=0`` or ``skip`` → no auto-run (call ``run_faithfulness()`` yourself).
    """
    raw = os.environ.get("GP_FAITH_AUTO", "").strip()
    if raw.lower() in ("0", "false", "no", "n", "skip"):
        return

    _ensure_faithfulness_deps(_default_backend())
    _try_load_hf_token()
    _apply_default_sifatali_rag_env()
    _apply_default_prior_faith_dirs()
    ctx_preview = (
        os.environ.get("GP_FAITH_CONTEXT", "").strip().lower()
        or _default_faith_context_mode()
    )
    if _faith_context_needs_reretrieve(ctx_preview):
        _ensure_rag_retrieval_deps()

    batch_size = _default_batch_size()
    if os.environ.get("GP_FAITH_BATCH_SIZE", "").strip().isdigit():
        batch_size = int(os.environ.get("GP_FAITH_BATCH_SIZE", "").strip())
    limit = int(os.environ.get("GP_FAITH_LIMIT", "0") or "0")
    no_resume = os.environ.get("GP_FAITH_NO_RESUME", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    ckpt_every = int(os.environ.get("GP_FAITH_CHECKPOINT_EVERY", "1") or "1")
    backend, model = _normalize_faithfulness_run(
        _default_backend(),
        os.environ.get("GP_FAITHFULNESS_MODEL", "").strip(),
    )
    resolved_model = _resolve_faithfulness_model(model)
    rag_only = os.environ.get("GP_FAITH_RAG_ONLY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    try:
        pred = _resolve_predictions_path(_default_predictions_path())
    except FileNotFoundError as exc:
        print(f"[faithfulness] {exc}", flush=True)
        print(
            "[faithfulness] Attach Add Data → fatinshadab/benchmark-results-all-predictions-combined, "
            "then re-run this cell.",
            flush=True,
        )
        return

    ctx_mode = os.environ.get("GP_FAITH_CONTEXT", "").strip().lower() or _default_faith_context_mode()
    _warn_faithfulness_context_setup(ctx_mode)
    print(
        f"[faithfulness] auto-run predictions={pred!r} batch_size={batch_size} "
        f"rag_only={rag_only} mcq_only={_default_mcq_only()} "
        f"strip_prompt={_default_strip_prompt()} "
        f"context_mode={ctx_mode} backend={backend} "
        f"model={resolved_model!r} resume={not no_resume}",
        flush=True,
    )
    try:
        run_faithfulness(
            pred,
            batch_size=batch_size,
            rag_only=rag_only,
            backend=backend,
            model=model,
            limit=limit,
            resume=not no_resume,
            checkpoint_every=ckpt_every,
        )
    except Exception as exc:
        print(f"[faithfulness] ERROR: {exc}", flush=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Faithfulness scoring (DeepEval+HF / OpenRouter)")
    p.add_argument("--input", default="", help="JSONL with context + answer per line")
    p.add_argument(
        "--predictions",
        default="",
        help=(
            "Predictions CSV or JSON (default on Kaggle: "
            f"{_KAGGLE_PREDICTIONS_CSV})"
        ),
    )
    p.add_argument("--out", default="", help="Write full results JSON (single-shot mode)")
    p.add_argument(
        "--backend",
        default=_default_backend(),
        choices=["auto", "openrouter", "deepeval", "local_hf"],
        help="Scoring backend (Kaggle default: local_hf + Qwen2.5-3B on GPU).",
    )
    p.add_argument(
        "--model",
        default="",
        help="HF model (Kaggle default: Qwen/Qwen2.5-3B-Instruct; else Qwen/Qwen3-8B).",
    )
    p.add_argument("--model-alt", default=os.environ.get("OPENROUTER_MODEL_ALT", ""))
    p.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help=f"Rows per run/batch CSV (default: {_DEFAULT_BATCH_SIZE} on Kaggle, 0 = single JSON).",
    )
    p.add_argument(
        "--rag_only",
        action="store_true",
        help="Only score rows with rag_flag=True (use --no_rag_only for Table 4 NoRAG comparability).",
    )
    p.add_argument(
        "--no_rag_only",
        action="store_true",
        help="Score all rows with context+answer (overrides default rag_only on batch runs).",
    )
    p.add_argument(
        "--norag_only",
        action="store_true",
        help="Score only rag_flag=False rows (Table 4 NoRAG grounding; no FAISS required).",
    )
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds between OpenRouter API calls")
    p.add_argument("--limit", type=int, default=0, help="Max rows (0 = all; single-shot or pre-batch cap)")
    p.add_argument(
        "--last_rows",
        type=int,
        default=-1,
        help=f"Keep only the last N raw CSV rows (default {_KAGGLE_LAST_ROWS} on Kaggle; 0 = full file).",
    )
    p.add_argument(
        "--row_offset",
        type=int,
        default=-1,
        help=f"Skip first N raw CSV rows (default {_KAGGLE_PREDICTIONS_ROW_OFFSET} on Kaggle; 0 = from start).",
    )
    p.add_argument(
        "--start_batch",
        type=int,
        default=-1,
        help=(
            f"1-indexed first batch (default {_KAGGLE_START_BATCH} on Kaggle = rows "
            f"{_KAGGLE_START_RAG_ROW}-{_TARGET_RAG_ROWS}; use 1 for full run)."
        ),
    )
    p.add_argument("--max_ctx_chars", type=int, default=-1, help="Truncate context/answer (default 6000)")
    p.add_argument("--no_resume", action="store_true", help="Ignore batch checkpoint")
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Flush batch checkpoint every N newly scored rows.",
    )
    p.add_argument(
        "--no_mcq_only",
        action="store_true",
        help="Score full model response (default: MCQ selected option only).",
    )
    p.add_argument(
        "--no_strip_prompt",
        action="store_true",
        help="Keep RAG instruction header in context (default: evidence blocks only).",
    )
    p.add_argument(
        "--faith_context",
        default="",
        choices=["", "stored", "reretrieve", "gold", "reretrieve_or_stored"],
        help="CONTEXT source: stored CSV, reretrieve from rag_index, gold for PubMedQA (default: auto).",
    )
    args = p.parse_args(_filtered_cli_args())

    batch_size = args.batch_size if args.batch_size >= 0 else _default_batch_size()
    max_ctx = args.max_ctx_chars if args.max_ctx_chars >= 0 else _default_max_ctx_chars()
    use_mcq = _default_mcq_only() and not args.no_mcq_only
    use_strip = _default_strip_prompt() and not args.no_strip_prompt
    use_ctx = (args.faith_context or _default_faith_context_mode()).strip()
    norag_only = args.norag_only or _default_norag_only()
    rag_only = args.rag_only
    if norag_only:
        rag_only = False
    elif batch_size > 0 and not args.no_rag_only and not args.rag_only:
        rag_only = True

    _load_openrouter_keys_from_notebook()

    if args.backend in ("openrouter", "auto") and batch_size <= 0:
        key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not key and args.backend == "openrouter":
            print(
                "OPENROUTER_API_KEY is not set.\n\n"
                "Kaggle: Add-ons → Secrets → OPENROUTER_API_KEY\n"
                "Or use: --backend deepeval",
                file=sys.stderr,
            )
            sys.exit(2)

    pred_path = (args.predictions or _default_predictions_path()).strip()
    input_path = (args.input or "").strip()

    fb, fm = _normalize_faithfulness_run(args.backend, args.model)

    if batch_size > 0:
        if not pred_path and not input_path:
            pred_path = _resolve_predictions_path("")
        elif pred_path:
            pred_path = _resolve_predictions_path(pred_path)
        else:
            raise ValueError("Batch mode requires --predictions CSV/JSON (not JSONL --input).")
        run_faithfulness_batch(
            pred_path,
            batch_size=batch_size,
            rag_only=rag_only,
            norag_only=norag_only,
            backend=fb,
            model_id=fm,
            openrouter_model=args.model or "",
            max_ctx_chars=max_ctx,
            row_limit=args.limit,
            row_offset=args.row_offset,
            last_rows=args.last_rows,
            start_batch=args.start_batch,
            resume=not args.no_resume,
            checkpoint_every=max(1, args.checkpoint_every),
            mcq_only=use_mcq,
            strip_prompt=use_strip,
            context_mode=use_ctx,
        )
        return

    if pred_path:
        pred_path = _resolve_predictions_path(pred_path)
        raw_rows, _ = load_predictions_rows(pred_path)
        resolved_last = _resolve_last_rows(args.last_rows)
        if resolved_last > 0:
            raw_rows = _slice_last_rows(raw_rows, resolved_last)
        resolved_offset = _resolve_row_offset(args.row_offset)
        if resolved_offset > 0:
            raw_rows = _slice_raw_rows(raw_rows, resolved_offset)
        rows = _rows_for_faithfulness(
            raw_rows,
            rag_only=rag_only,
            mcq_only=use_mcq,
            strip_prompt=use_strip,
            context_mode=use_ctx,
        )
    elif input_path:
        rows = load_jsonl(input_path)
    else:
        rows = [dict(r) for r in _BUNDLED_SAMPLE_ROWS]
        print(f"No --input/--predictions; using {len(rows)} in-code sample rows.", flush=True)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("No rows to score.", file=sys.stderr)
        sys.exit(3)

    out = run_faithfulness_eval(
        rows,
        out_path=(args.out or "").strip(),
        backend=fb,
        model=fm,
        model_alt=args.model_alt,
        sleep_s=args.sleep,
        max_ctx_chars=max_ctx,
        mcq_only=use_mcq,
        context_mode=use_ctx,
    )
    if out.get("rows"):
        _print_batch_grounding_summary(out["rows"])
    if not (args.out or "").strip():
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
        print("(pass --out PATH.json to save; used by run_paper_eval_suite.py)", flush=True)


def _is_faithfulness_batch_csv(filename: str) -> bool:
    """True for ``*_batch_NNNN.csv`` outputs, not merged ``*_combined.csv`` files."""
    b = os.path.basename(filename).lower()
    return "_batch_" in b and b.endswith(".csv") and not b.endswith("_combined.csv")


def combine_faithfulness_batches(
    input_dir: str,
    output_path: str = "",
    *,
    batch_glob: str = "*_batch_*.csv",
) -> str:
    """
    Merge batch faithfulness CSVs in *input_dir* into one file (union of all columns).

    Excludes files whose names contain ``_combined``. Returns the output path written.
    """
    in_dir = os.path.abspath(input_dir)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Directory not found: {in_dir!r}")

    paths = sorted(
        p
        for p in glob.glob(os.path.join(in_dir, batch_glob))
        if _is_faithfulness_batch_csv(p)
    )
    if not paths:
        raise FileNotFoundError(
            f"No batch CSVs matching {batch_glob!r} under {in_dir!r}"
        )

    fieldnames: List[str] = []
    seen_cols: set = set()
    rows_out: List[Dict[str, Any]] = []

    for path in paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for col in reader.fieldnames:
                if col not in seen_cols:
                    seen_cols.add(col)
                    fieldnames.append(col)
            for row in reader:
                rows_out.append(
                    {k: (v if v is not None else "") for k, v in row.items()}
                )

    if not fieldnames and rows_out:
        fieldnames = sorted({k for r in rows_out for k in r})

    if not output_path.strip():
        stem = os.path.commonprefix([os.path.basename(p) for p in paths])
        stem = stem.split("_batch_")[0] or "faithfulness"
        output_path = os.path.join(in_dir, f"{stem}_combined.csv")
    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({c: row.get(c, "") for c in fieldnames})

    print(
        f"[faithfulness] Combined {len(paths)} batches -> {out_path!r} ({len(rows_out)} rows)",
        flush=True,
    )
    return out_path


def combine_faithfulness_dir(input_dir: str) -> List[str]:
    """Combine each batch series (by filename stem) under *input_dir*."""
    in_dir = os.path.abspath(input_dir)
    stems: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "*_batch_*.csv"))):
        if not _is_faithfulness_batch_csv(path):
            continue
        base = os.path.basename(path)
        stem = base.rsplit("_batch_", 1)[0]
        stems.setdefault(stem, []).append(path)

    written: List[str] = []
    for stem in sorted(stems):
        out = combine_faithfulness_batches(
            in_dir,
            os.path.join(in_dir, f"{stem}_combined.csv"),
            batch_glob=f"{stem}_batch_*.csv",
        )
        written.append(out)
    return written


def _faith_csv_dedupe_key(row: Dict[str, Any]) -> str:
    """One faithfulness score per prediction row (matches scoring checkpoint key)."""
    return _row_key(row)


def dedupe_faithfulness_csv(
    input_path: str,
    output_path: str = "",
    *,
    keep: str = "last",
) -> str:
    """
    Remove duplicate rows from a faithfulness CSV (same id/question_id/model/benchmark).

    *keep*: ``last`` (default) or ``first``.
    """
    in_path = os.path.abspath(input_path)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    rows_in: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    with open(in_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_in.append(row)

    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows_in:
        k = _faith_csv_dedupe_key(row)
        if k not in by_key:
            order.append(k)
        if keep == "first" and k in by_key:
            continue
        by_key[k] = row

    rows_out = [by_key[k] for k in order]
    if not output_path.strip():
        base, ext = os.path.splitext(in_path)
        if base.endswith("_all_combined"):
            output_path = base.replace("_all_combined", "_all_deduped") + ext
        elif base.endswith("_combined"):
            output_path = base.replace("_combined", "_deduped") + ext
        else:
            output_path = base + "_deduped" + ext
    out_path = os.path.abspath(output_path)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({c: row.get(c, "") for c in fieldnames})

    removed = len(rows_in) - len(rows_out)
    print(
        f"[faithfulness] Deduped {in_path!r} -> {out_path!r}: "
        f"{len(rows_out)} rows kept, {removed} duplicates removed",
        flush=True,
    )
    return out_path


def _faithfulness_target_rows() -> int:
    """Expected row count for deduped final (env ``GP_FAITH_TARGET_ROWS``: rag | norag | all | int)."""
    raw = os.environ.get("GP_FAITH_TARGET_ROWS", "").strip().lower()
    if raw in ("all", "full", "36360"):
        return _TARGET_ALL_ROWS
    if raw in ("norag", "18180_norag"):
        return _TARGET_NORAG_ROWS
    if raw in ("rag", "18180", "18180_rag"):
        return _TARGET_RAG_ROWS
    try:
        n = int(raw)
        if n > 0:
            return n
    except ValueError:
        pass
    return _TARGET_RAG_ROWS


def build_deduped_faithfulness_final(
    input_dir: str,
    output_path: str = "",
) -> str:
    """
    Build one deduplicated faithfulness CSV from all batch CSVs under *input_dir*.

    Later batch files overwrite earlier rows with the same ``_row_key``.
    """
    in_dir = os.path.abspath(input_dir)
    batch_paths = sorted(
        p
        for p in glob.glob(os.path.join(in_dir, "**", "*_batch_*.csv"), recursive=True)
        if _is_faithfulness_batch_csv(p)
    )
    if not batch_paths:
        for name in (
            "benchmark_results_all_predictions_combined_faithfulness_combined.csv",
            "benchmark_results_all_predictions_combined_faithfulness_from12500_combined.csv",
            "benchmark_results_all_predictions_combined_faithfulness_all_combined.csv",
        ):
            p = os.path.join(in_dir, name)
            if os.path.isfile(p):
                batch_paths = [p]
                break
    if not batch_paths:
        raise FileNotFoundError(
            f"No faithfulness batch CSVs under {in_dir!r}. Run scoring batches first."
        )

    by_key: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    fieldnames: List[str] = []
    n_read = 0

    for path in batch_paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for col in reader.fieldnames:
                    if col not in fieldnames:
                        fieldnames.append(col)
            for row in reader:
                n_read += 1
                try:
                    v = float(row.get("faithfulness_primary", ""))
                    if v != v:
                        continue
                except (TypeError, ValueError):
                    continue
                k = _faith_csv_dedupe_key(row)
                if k not in by_key:
                    order.append(k)
                by_key[k] = row

    if not output_path.strip():
        output_path = os.path.join(
            in_dir,
            "benchmark_results_all_predictions_combined_faithfulness_final.csv",
        )
    out_path = os.path.abspath(output_path)
    rows_out = [by_key[k] for k in order]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_out:
            writer.writerow({c: row.get(c, "") for c in fieldnames})

    target = _faithfulness_target_rows()
    gap = target - len(rows_out)
    print(
        f"[faithfulness] Final deduped file: {out_path!r} "
        f"({len(rows_out)} unique rows from {n_read} scored lines in {len(batch_paths)} files; "
        f"target {target}; "
        f"{'complete' if gap == 0 else f'missing {gap} rows'})",
        flush=True,
    )
    return out_path


_PAPER_DEFAULT_MODEL = "Qwen/Qwen3-8B"
_PAPER_KAGGLE_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _resolve_paper_model() -> str:
    explicit = os.environ.get("GP_FAITH_PAPER_MODEL", "").strip()
    if explicit:
        return explicit
    if os.path.isdir("/kaggle"):
        return _PAPER_KAGGLE_DEFAULT_MODEL
    return _PAPER_DEFAULT_MODEL


def _pin_fatinshadab_data_paths() -> Tuple[str, str]:
    """
    Pin fatinshadab Kaggle paths: predictions CSV + ``rag_index/`` + ``rag_index_gold/``.
    """
    os.environ.setdefault("GP_FAITH_ACCOUNT", "fatinshadab")
    os.environ.setdefault("GP_FAITH_PREDICTIONS", _KAGGLE_PREDICTIONS_CSV)
    primary = _discover_rag_index_dir(_KAGGLE_RAG_INDEX_FATINSHADAB, gold=False)
    gold = _discover_rag_index_dir(_KAGGLE_RAG_GOLD_INDEX_FATINSHADAB, gold=True)
    if primary:
        os.environ["GP_FAITH_RAG_INDEX_DIR"] = primary
    else:
        os.environ.setdefault("GP_FAITH_RAG_INDEX_DIR", _KAGGLE_RAG_INDEX_FATINSHADAB)
    if gold:
        os.environ["GP_FAITH_RAG_GOLD_DIR"] = gold
    else:
        os.environ.setdefault("GP_FAITH_RAG_GOLD_DIR", _KAGGLE_RAG_GOLD_INDEX_FATINSHADAB)
    return primary or _KAGGLE_RAG_INDEX_FATINSHADAB, gold or _KAGGLE_RAG_GOLD_INDEX_FATINSHADAB


def _paper_faith_output_root(subdir: str = "paper") -> str:
    if os.path.isdir("/kaggle"):
        return os.path.join(_KAGGLE_FAITH_WRITE_DIR, subdir)
    return os.path.join(_script_dir(), "Faithfulness", subdir)


def apply_paper_run_env(*, rescore_all: bool = False, out_subdir: str = "paper") -> str:
    """
    Configure env for paper-quality faithfulness: re-retrieved context + strong local judge.

    Uses a separate output dir so prior ``fathfullness/`` (stored + 3B) batches are not treated
    as already scored. Set ``GP_FAITH_PAPER_MODEL`` to override the judge (default Qwen3-8B).
    """
    model = _resolve_paper_model()
    paper_root = _paper_faith_output_root(out_subdir)
    os.makedirs(paper_root, exist_ok=True)

    os.environ["GP_FAITH_CONTEXT"] = "reretrieve_or_stored"
    os.environ["GP_FAITHFULNESS_BACKEND"] = "local_hf"
    os.environ["GP_FAITHFULNESS_MODEL"] = model
    os.environ["GP_FAITH_STRICT"] = "1"
    os.environ["GP_FAITH_BEST_CONTEXT"] = "1"
    os.environ["GP_FAITH_MEDQA_BLEND"] = "1"
    os.environ["GP_FAITH_CALIBRATE"] = "0"
    os.environ["GP_FAITH_OUT_DIR"] = paper_root
    os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = paper_root
    os.environ["GP_FAITH_ROW_OFFSET"] = "0"
    os.environ["GP_FAITH_LAST_ROWS"] = "0"
    if os.path.isdir("/kaggle"):
        os.environ.setdefault("GP_FAITH_USE_4BIT", "1")
        _ensure_bitsandbytes()
    acct = os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower()
    if acct in ("", "fatinshadab"):
        primary, gold = _pin_fatinshadab_data_paths()
    else:
        primary = _discover_rag_index_dir(
            os.environ.get("GP_FAITH_RAG_INDEX_DIR", "").strip()
            or _KAGGLE_RAG_DATASET_PRIMARY,
            gold=False,
        )
        gold = _discover_rag_index_dir(
            os.environ.get("GP_FAITH_RAG_GOLD_DIR", "").strip()
            or _KAGGLE_RAG_DATASET_GOLD,
            gold=True,
        )
        if primary:
            os.environ.setdefault("GP_FAITH_RAG_INDEX_DIR", primary)
        if gold:
            os.environ.setdefault("GP_FAITH_RAG_GOLD_DIR", gold)
    _log_pinned_rag_paths(
        os.environ.get("GP_FAITH_RAG_INDEX_DIR", ""),
        os.environ.get("GP_FAITH_RAG_GOLD_DIR", ""),
    )

    if rescore_all:
        os.environ["GP_FAITH_ONLY_MISSING"] = "0"
        os.environ["GP_FAITH_START_BATCH"] = "1"
    else:
        os.environ.setdefault("GP_FAITH_ONLY_MISSING", "1")
        # Do not set GP_FAITH_START_BATCH here — sifatali resume / full_coverage set it explicitly.

    print(
        "[faithfulness] Paper run env: "
        f"context=reretrieve_or_stored model={model!r} out_dir={paper_root!r} "
        f"only_missing={os.environ.get('GP_FAITH_ONLY_MISSING', '1')} "
        f"start_batch={os.environ.get('GP_FAITH_START_BATCH', '(from _default_start_batch)')}",
        flush=True,
    )
    return paper_root


def apply_norag_paper_run_env(*, prior_rag_dirs: str = "") -> str:
    """
    Configure env for **NoRAG-only** Table~4 faithfulness (task abstract / MCQ vignette grounding).

    Does not require RAG indexes. Writes batches under ``Faithfulness/norag/``.
    Attach prior RAG faithfulness batches as a dataset and pass *prior_rag_dirs* so
    ``GP_FAITH_ONLY_MISSING`` can merge coverage when rebuilding the full 36,360-row final.
    """
    norag_root = _paper_faith_output_root("norag")
    os.makedirs(norag_root, exist_ok=True)
    model = _resolve_faithfulness_model(os.environ.get("GP_FAITHFULNESS_MODEL", ""))
    os.environ["GP_FAITH_CONTEXT"] = "stored"
    os.environ["GP_FAITH_NORAG_ONLY"] = "1"
    os.environ["GP_FAITH_RAG_ONLY"] = "0"
    os.environ["GP_FAITHFULNESS_BACKEND"] = "local_hf"
    os.environ["GP_FAITHFULNESS_MODEL"] = model
    os.environ["GP_FAITH_OUT_DIR"] = norag_root
    os.environ["GP_FAITH_START_BATCH"] = "1"
    os.environ.setdefault("GP_FAITH_ONLY_MISSING", "1")
    os.environ["GP_FAITH_ROW_OFFSET"] = "0"
    os.environ["GP_FAITH_LAST_ROWS"] = "0"
    prior_parts = [p for p in (norag_root, (prior_rag_dirs or "").strip()) if p]
    os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = ",".join(prior_parts)
    if os.path.isdir("/kaggle"):
        os.environ.setdefault("GP_FAITH_USE_4BIT", "1")
        _ensure_bitsandbytes()
        _pin_fatinshadab_data_paths()
    print(
        f"[faithfulness] NoRAG paper env: context=task_grounding model={model!r} "
        f"out_dir={norag_root!r} target={_TARGET_NORAG_ROWS} NoRAG rows",
        flush=True,
    )
    return norag_root


def run_faithfulness_norag_paper_run(
    predictions_path: str = "",
    *,
    prior_rag_dirs: str = "",
    batch_size: int = -1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Score **NoRAG-only** rows for Table~4 (comparable grounding vs RAG).

    Re-run the notebook cell until all ~37 batches complete, then merge with RAG CSV::

        os.environ["GP_FAITH_TARGET_ROWS"] = "all"
        build_deduped_faithfulness_final("/kaggle/working/Faithfulness")
    """
    apply_norag_paper_run_env(prior_rag_dirs=prior_rag_dirs)
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _KAGGLE_PREDICTIONS_CSV
    )
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=False,
        norag_only=True,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", ""),
        resume=resume,
        checkpoint_every=1,
    )


def apply_account_paper_run_env(
    *,
    account: str,
    predictions_csv: str,
    rag_primary: str,
    rag_gold: str,
    start_batch: int,
    prior_batch_dirs: str = "",
    out_subdir: str = "paper",
    prior_batch_hint: str = "",
) -> str:
    """Configure paper faithfulness for a Kaggle account + resume from *start_batch*."""
    os.environ["GP_FAITH_ACCOUNT"] = account
    os.environ["GP_FAITH_PREDICTIONS"] = predictions_csv
    os.environ["GP_FAITH_START_BATCH"] = str(max(1, int(start_batch)))
    os.environ["GP_FAITH_ONLY_MISSING"] = "1"
    os.environ["GP_FAITH_ROW_OFFSET"] = "0"
    os.environ["GP_FAITH_LAST_ROWS"] = "0"

    primary = _discover_rag_index_dir(rag_primary, gold=False)
    gold = _discover_rag_index_dir(rag_gold, gold=True)
    if primary:
        os.environ["GP_FAITH_RAG_INDEX_DIR"] = primary
    else:
        os.environ.setdefault("GP_FAITH_RAG_INDEX_DIR", rag_primary)
    if gold:
        os.environ["GP_FAITH_RAG_GOLD_DIR"] = gold
    else:
        os.environ.setdefault("GP_FAITH_RAG_GOLD_DIR", rag_gold)

    paper_root = apply_paper_run_env(rescore_all=False, out_subdir=out_subdir)

    os.environ["GP_FAITH_ACCOUNT"] = account
    os.environ["GP_FAITH_PREDICTIONS"] = predictions_csv
    os.environ["GP_FAITH_START_BATCH"] = str(max(1, int(start_batch)))

    priors: List[str] = [paper_root]
    for part in (prior_batch_dirs or "").split(","):
        p = part.strip()
        if p and p not in priors:
            priors.append(p)
    os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = ",".join(priors)

    row_start = (max(1, int(start_batch)) - 1) * _DEFAULT_BATCH_SIZE + 1
    last_prior_idx = max(0, int(start_batch) - 2)
    print(
        f"[faithfulness] {account} paper resume: "
        f"predictions={predictions_csv!r} "
        f"primary={os.environ.get('GP_FAITH_RAG_INDEX_DIR', '')!r} "
        f"gold={os.environ.get('GP_FAITH_RAG_GOLD_DIR', '')!r} "
        f"start_batch={start_batch}/37 (rows {row_start}-18180) "
        f"prior_dirs={os.environ.get('GP_FAITH_PRIOR_BATCH_DIRS', '')!r}",
        flush=True,
    )
    if not prior_batch_dirs.strip():
        hint = prior_batch_hint or (
            f"Attach batch_0000–batch_{last_prior_idx:04d}, then set prior_batch_dirs='...'"
        )
        print(f"[faithfulness] {hint}", flush=True)
    return paper_root


def apply_sifatali_paper_run_env(
    *,
    start_batch: int = _SIFATALI_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    out_subdir: str = "paper",
) -> str:
    """
    Paper run on **sifatali008** account: flat ``rag-index/`` + resume from batch *start_batch*.

    Attach batches ``*_batch_0000`` … ``*_batch_0010`` from the other account via
    *prior_batch_dirs* (Kaggle dataset path). Rows **5,001–18,180** when ``start_batch=11``.
    """
    return apply_account_paper_run_env(
        account="sifatali008",
        predictions_csv=_SIFATALI_PREDICTIONS_CSV,
        rag_primary=_KAGGLE_RAG_INDEX_SIFATALI,
        rag_gold=_KAGGLE_RAG_GOLD_SIFATALI,
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
        out_subdir=out_subdir,
        prior_batch_hint=(
            "Attach batches 0000-0010 from the other account, then set:\n"
            "  prior_batch_dirs='/kaggle/input/YOUR_DATASET/path_to_batches'"
        ),
    )


def apply_fahim220_paper_run_env(
    *,
    start_batch: int = _FAHIM220_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    out_subdir: str = "paper",
) -> str:
    """
    Paper run on **fahim220** account: flat ``rag-index/`` + resume from batch *start_batch*.

    Rows **7,501–18,180** when ``start_batch=16`` (attach ``batch_0000`` … ``batch_0015``).
    """
    return apply_account_paper_run_env(
        account="fahim220",
        predictions_csv=_FAHIM220_PREDICTIONS_CSV,
        rag_primary=_KAGGLE_RAG_INDEX_FAHIM220,
        rag_gold=_KAGGLE_RAG_GOLD_FAHIM220,
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
        out_subdir=out_subdir,
        prior_batch_hint=(
            "Attach batches 0000-0015 from prior runs, then set:\n"
            "  prior_batch_dirs='/kaggle/input/YOUR_DATASET/path_to_batches'"
        ),
    )


def apply_fatinshadab_paper_run_env(
    *,
    start_batch: int = _FATINSHADAB_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    out_subdir: str = "paper",
) -> str:
    """
    Paper run on **fatinshadab**: ``rag_index/`` + ``rag_index_gold/`` + combined predictions CSV.

    Default ``start_batch=11`` resumes rows 5,001–18,180 (attach ``batch_0000`` … ``batch_0010``).
    Use ``start_batch=1`` for a fresh full run.
    """
    return apply_account_paper_run_env(
        account="fatinshadab",
        predictions_csv=_KAGGLE_PREDICTIONS_CSV,
        rag_primary=_KAGGLE_RAG_INDEX_FATINSHADAB,
        rag_gold=_KAGGLE_RAG_GOLD_INDEX_FATINSHADAB,
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
        out_subdir=out_subdir,
        prior_batch_hint=(
            "Attach batches 0000-0010 from prior runs, then set:\n"
            "  prior_batch_dirs='/kaggle/input/YOUR_DATASET/path_to_batches'"
        ),
    )


def apply_ummesalmahabiba_paper_run_env(
    *,
    start_batch: int = _UMMESALMAHABIBA_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    out_subdir: str = "paper",
) -> str:
    """
    Paper run on **ummesalmahabiba**: ``rag_index/`` + ``rag_index_gold/`` subfolders.

    Default ``start_batch=36`` resumes rows 17,501–18,180 (attach ``batch_0000`` … ``batch_0035``).
    """
    return apply_account_paper_run_env(
        account="ummesalmahabiba",
        predictions_csv=_UMMESALMAHABIBA_PREDICTIONS_CSV,
        rag_primary=_KAGGLE_RAG_INDEX_UMMESALMAHABIBA,
        rag_gold=_KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA,
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
        out_subdir=out_subdir,
        prior_batch_hint=(
            "Attach batches 0000-0035 from prior runs, then set:\n"
            "  prior_batch_dirs='/kaggle/input/YOUR_DATASET/path_to_batches'"
        ),
    )


def apply_gap_fill_run_env(
    *,
    missing_predictions_csv: str = "",
    gap_out_dir: str = "",
    prior_rerun_dir: str = "",
    account: str = "ummesalmahabiba",
    start_batch: int = 1,
) -> str:
    """
    Score ``missing_rag_predictions.csv`` only (strict prior keys).

    *gap_out_dir* must be a **directory** under ``/kaggle/working/``, not the CSV path.

    Default account **ummesalmahabiba** (last 2,500 / 4,785 gap rows, 5 batches)::

        predictions: .../ummesalmahabiba/missing-rag-predictions/missing_rag_predictions.csv
        RAG:         .../ummesalmahabiba/rag-index/rag_index/  (+ rag-index-gold)
    """
    acct = (account or "ummesalmahabiba").strip().lower()
    if acct == "ummesalmahabiba":
        missing = (missing_predictions_csv or "").strip() or _UMMESALMAHABIBA_GAP_FILL_PREDICTIONS
        out_default = _UMMESALMAHABIBA_GAP_FILL_OUT_DIR
    elif acct == "fahim220":
        missing = (missing_predictions_csv or "").strip() or _FAHIM220_GAP_FILL_PREDICTIONS
        out_default = _FAHIM220_GAP_FILL_OUT_DIR
    elif acct == "fatinshadab":
        missing = (missing_predictions_csv or "").strip() or ""
        out_default = "/kaggle/working/fathfullness/rerun/gap_fill"
    else:
        missing = (missing_predictions_csv or "").strip()
        out_default = "/kaggle/working/fathfullness/rerun/gap_fill"
    if not missing:
        missing = "/kaggle/working/missing_rag_predictions.csv"
    out_dir = (gap_out_dir or "").strip() or out_default
    if out_dir.lower().endswith(".csv") or os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir) or "/kaggle/working/fathfullness/rerun/gap_fill"

    os.makedirs(out_dir, exist_ok=True)
    os.environ["GP_FAITH_GAP_FILL"] = "1"
    if prior_rerun_dir.strip().lower().endswith(".csv"):
        print(
            f"[faithfulness] prior_rerun_dir looks like a CSV file ({prior_rerun_dir!r}); "
            "use a folder containing batch_*.csv, not the predictions path.",
            flush=True,
        )
    # Paper env + RAG paths only — do not reset predictions to full 18k CSV
    apply_paper_run_env(rescore_all=False, out_subdir="gap_fill")
    if acct == "ummesalmahabiba":
        os.environ["GP_FAITH_ACCOUNT"] = "ummesalmahabiba"
        primary = _discover_rag_index_dir(_KAGGLE_RAG_INDEX_UMMESALMAHABIBA, gold=False)
        gold = _discover_rag_index_dir(_KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA, gold=True)
        if not primary:
            os.environ.setdefault("GP_FAITH_RAG_INDEX_DIR", _KAGGLE_RAG_INDEX_UMMESALMAHABIBA)
            primary = _rag_dir_from_env("GP_FAITH_RAG_INDEX_DIR") or _KAGGLE_RAG_INDEX_UMMESALMAHABIBA
        if not gold:
            os.environ.setdefault("GP_FAITH_RAG_GOLD_DIR", _KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA)
            gold = _rag_dir_from_env("GP_FAITH_RAG_GOLD_DIR") or _KAGGLE_RAG_GOLD_INDEX_UMMESALMAHABIBA
    elif acct == "fahim220":
        os.environ["GP_FAITH_ACCOUNT"] = "fahim220"
        primary = _discover_rag_index_dir(_KAGGLE_RAG_INDEX_FAHIM220, gold=False)
        gold = _discover_rag_index_dir(_KAGGLE_RAG_GOLD_FAHIM220, gold=True)
        if not primary:
            os.environ.setdefault("GP_FAITH_RAG_INDEX_DIR", _KAGGLE_RAG_INDEX_FAHIM220)
            primary = _rag_dir_from_env("GP_FAITH_RAG_INDEX_DIR") or _KAGGLE_RAG_INDEX_FAHIM220
        if not gold:
            os.environ.setdefault("GP_FAITH_RAG_GOLD_DIR", _KAGGLE_RAG_GOLD_FAHIM220)
            gold = _rag_dir_from_env("GP_FAITH_RAG_GOLD_DIR") or _KAGGLE_RAG_GOLD_FAHIM220
    elif acct == "fatinshadab":
        os.environ["GP_FAITH_ACCOUNT"] = "fatinshadab"
        primary, gold = _pin_fatinshadab_data_paths()
    else:
        os.environ["GP_FAITH_ACCOUNT"] = acct
        primary, gold = "", ""
    if primary:
        os.environ["GP_FAITH_RAG_INDEX_DIR"] = primary
    if gold:
        os.environ["GP_FAITH_RAG_GOLD_DIR"] = gold

    os.environ["GP_FAITH_PREDICTIONS"] = missing
    os.environ["GP_FAITH_OUT_DIR"] = out_dir
    os.environ["GP_FAITH_START_BATCH"] = str(max(1, int(start_batch)))
    os.environ["GP_FAITH_ONLY_MISSING"] = "1"
    os.environ["GP_FAITH_PRIOR_MATCH_STRICT"] = "1"
    os.environ["GP_FAITH_CONTEXT"] = "reretrieve_or_stored"
    if os.path.isdir("/kaggle"):
        os.environ.setdefault("GP_FAITH_USE_4BIT", "1")
        os.environ.setdefault("GP_FAITHFULNESS_MODEL", _FALLBACK_FAITHFULNESS_MODEL)
    priors: List[str] = [out_dir]
    if prior_rerun_dir.strip():
        priors.append(prior_rerun_dir.strip())
    os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = ",".join(priors)

    if os.path.isdir("/kaggle"):
        _ensure_rag_retrieval_deps()
    primary = _resolve_faith_rag_primary_dir()
    gold = _resolve_faith_rag_gold_dir()
    print(
        f"[faithfulness] Gap-fill env: predictions={missing!r} out_dir={out_dir!r} "
        f"start_batch={os.environ.get('GP_FAITH_START_BATCH', '1')} "
        f"primary={primary!r} gold={gold!r}",
        flush=True,
    )
    if not primary:
        print(
            "[faithfulness] Attach rag-index + rag-index-gold to this notebook (Add Data). "
            "Gap-fill will use stored CSV context until indexes are available.",
            flush=True,
        )
        os.environ["GP_FAITH_ALLOW_STORED_FALLBACK"] = "1"
    return out_dir


def apply_ummesalmahabiba_gap_fill_run_env(
    *,
    start_batch: int = _UMMESALMAHABIBA_GAP_FILL_START_BATCH,
    prior_rerun_dir: str = "",
    gap_out_dir: str = "",
) -> str:
    """
    Gap-fill on ummesalmahabiba — default **batch 6/10** on the full 4,785-row CSV.

    Batches 1–5 (rows 1–2,500) assumed done on another account; this run does 6–10
    (rows 2,501–4,785). If your CSV has only 2,500 rows, pass ``start_batch=1``.
    """
    return apply_gap_fill_run_env(
        missing_predictions_csv=_UMMESALMAHABIBA_GAP_FILL_PREDICTIONS,
        gap_out_dir=gap_out_dir or _UMMESALMAHABIBA_GAP_FILL_OUT_DIR,
        prior_rerun_dir=prior_rerun_dir,
        account="ummesalmahabiba",
        start_batch=start_batch,
    )


def run_faithfulness_ummesalmahabiba_gap_fill(
    *,
    start_batch: int = _UMMESALMAHABIBA_GAP_FILL_START_BATCH,
    prior_rerun_dir: str = "",
    resume: bool = True,
) -> Dict[str, Any]:
    """Score missing rows on ummesalmahabiba (default from batch 6/10 on 4,785-row file)."""
    apply_ummesalmahabiba_gap_fill_run_env(
        start_batch=start_batch,
        prior_rerun_dir=prior_rerun_dir,
    )
    return run_faithfulness(
        _UMMESALMAHABIBA_GAP_FILL_PREDICTIONS,
        rag_only=True,
        backend="local_hf",
        start_batch=start_batch,
        resume=resume,
        checkpoint_every=1,
    )


def apply_fahim220_gap_fill_run_env(
    *,
    prior_rerun_dir: str = "",
    gap_out_dir: str = "",
) -> str:
    """Gap-fill on **fahim220** paths (missing CSV + flat ``rag-index/``)."""
    return apply_gap_fill_run_env(
        missing_predictions_csv=_FAHIM220_GAP_FILL_PREDICTIONS,
        gap_out_dir=gap_out_dir or _FAHIM220_GAP_FILL_OUT_DIR,
        prior_rerun_dir=prior_rerun_dir,
        account="fahim220",
    )


def run_faithfulness_fahim220_gap_fill(
    *,
    prior_rerun_dir: str = "",
    resume: bool = True,
) -> Dict[str, Any]:
    """Score 4,785 missing rows using fahim220 Kaggle datasets."""
    apply_fahim220_gap_fill_run_env(prior_rerun_dir=prior_rerun_dir)
    return run_faithfulness(
        _FAHIM220_GAP_FILL_PREDICTIONS,
        rag_only=True,
        backend="local_hf",
        resume=resume,
        checkpoint_every=1,
    )


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _load_prior_faith_scores_by_key(csv_path: str) -> Dict[str, float]:
    """Map ``_row_key`` -> faithfulness_primary from an existing faithfulness CSV."""
    out: Dict[str, float] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row.get("faithfulness_primary", ""))
                if v != v:
                    continue
            except (TypeError, ValueError):
                continue
            out[_faith_csv_dedupe_key(row)] = v
    return out


def _stratified_sample_rows(
    rows: List[Dict[str, Any]],
    n_total: int,
    *,
    seed: int = 42,
    stratum_key: str = "benchmark",
) -> List[Dict[str, Any]]:
    if n_total <= 0 or not rows:
        return []
    if n_total >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_stratum: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_stratum.setdefault(str(row.get(stratum_key) or "unknown"), []).append(row)
    strata = sorted(by_stratum)
    base = n_total // len(strata)
    extra = n_total % len(strata)
    picked: List[Dict[str, Any]] = []
    for i, s in enumerate(strata):
        pool = by_stratum[s]
        k = min(len(pool), base + (1 if i < extra else 0))
        if k > 0:
            picked.extend(rng.sample(pool, k))
    if len(picked) < n_total:
        remaining = [r for r in rows if r not in picked]
        need = n_total - len(picked)
        if remaining and need > 0:
            picked.extend(rng.sample(remaining, min(need, len(remaining))))
    return picked[:n_total]


def _summarize_validation_pairs(
    label_a: str,
    scores_a: List[float],
    label_b: str,
    scores_b: List[float],
    *,
    tolerance: float = 10.0,
) -> None:
    pairs = [(a, b) for a, b in zip(scores_a, scores_b) if a == a and b == b]
    if len(pairs) < 2:
        print(
            f"[faithfulness] Validation {label_a} vs {label_b}: insufficient pairs",
            flush=True,
        )
        return
    xs, ys = zip(*pairs)
    r = _pearson(list(xs), list(ys))
    mae = statistics.mean(abs(a - b) for a, b in pairs)
    within = 100.0 * sum(1 for a, b in pairs if abs(a - b) <= tolerance) / len(pairs)
    print(
        f"[faithfulness] Validation {label_a} vs {label_b}: "
        f"n={len(pairs)} pearson={r:.3f} MAE={mae:.1f} "
        f"pct_within_{int(tolerance)}pt={within:.1f}%",
        flush=True,
    )


def run_deepeval_validation(
    predictions_path: str = "",
    *,
    prior_faithfulness_csv: str = "",
    sample_size: int = -1,
    seed: int = -1,
    output_path: str = "",
    paper_model: str = "",
) -> str:
    """
    DeepEval ``FaithfulnessMetric`` on a stratified RAG sample (no human audit).

    For each row: re-resolved context (paper settings), ``faithfulness_paper_hf`` (local HF),
    ``faithfulness_deepeval`` (DeepEval), and optional ``faithfulness_prior`` from an old CSV.

    Use for a methods footnote: agreement between automated metrics on a fixed subset.
    """
    os.environ["GP_FAITH_VALIDATION_DEEPEVAL"] = "1"
    apply_paper_run_env(rescore_all=False)

    n_sample = sample_size
    if n_sample < 0:
        env_n = os.environ.get("GP_FAITH_VALIDATION_N", "500").strip()
        try:
            n_sample = max(1, int(env_n))
        except ValueError:
            n_sample = 500

    val_seed = seed
    if val_seed < 0:
        env_s = os.environ.get("GP_FAITH_VALIDATION_SEED", "42").strip()
        try:
            val_seed = int(env_s)
        except ValueError:
            val_seed = 42

    predictions_path = _resolve_predictions_path(predictions_path)
    raw_rows, _ = load_predictions_rows(predictions_path)
    work_rows = _rows_for_faithfulness(
        raw_rows,
        rag_only=True,
        context_mode="reretrieve_or_stored",
    )
    if not work_rows:
        raise RuntimeError("No RAG rows available for validation sampling.")

    prior_path = prior_faithfulness_csv.strip()
    if not prior_path:
        for candidate in (
            os.path.join(_script_dir(), "fathfullness", "benchmark_results_all_predictions_combined_faithfulness_final.csv"),
            os.path.join(_script_dir(), "fathfullness", "benchmark_results_all_predictions_combined_faithfulness_all_combined.csv"),
        ):
            if os.path.isfile(candidate):
                prior_path = candidate
                break
    prior_by_key = _load_prior_faith_scores_by_key(prior_path)
    if prior_path:
        print(
            f"[faithfulness] Validation prior scores: {prior_path!r} ({len(prior_by_key)} keys)",
            flush=True,
        )

    sample = _stratified_sample_rows(work_rows, n_sample, seed=val_seed)
    print(
        f"[faithfulness] DeepEval validation sample: {len(sample)} rows "
        f"(stratified by benchmark, seed={val_seed})",
        flush=True,
    )

    mid = (paper_model or "").strip() or os.environ.get("GP_FAITH_PAPER_MODEL", _PAPER_DEFAULT_MODEL)
    ctx_mode = "reretrieve_or_stored"
    _warn_faithfulness_context_setup(ctx_mode)
    _init_faithfulness_rag(ctx_mode)
    _try_load_hf_token()
    _get_local_hf_model(mid)
    deepeval_metric = _get_deepeval_metric(mid)

    if not output_path.strip():
        out_dir = os.path.join(_script_dir(), "fathfullness")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(
            out_dir,
            f"faithfulness_deepeval_validation_n{len(sample)}.csv",
        )
    out_path = os.path.abspath(output_path)

    results: List[Dict[str, Any]] = []
    paper_scores: List[float] = []

    for i, mapped in enumerate(sample):
        src = mapped.get("_source_row") or {}
        ctx, ctx_src = _resolve_scoring_context(
            mapped, ctx_mode, strip_prompt=_default_strip_prompt()
        )
        q = str(mapped.get("question", "")).strip()
        ans = str(mapped.get("answer", "")).strip()
        rk = _row_key(mapped)
        prior_v = prior_by_key.get(rk, float("nan"))

        paper_s, paper_note = (
            (float("nan"), "empty_context_or_answer")
            if not ctx or not ans
            else score_one_local_hf(
                q, ctx, ans, model_id=mid, mcq_only=_default_mcq_only()
            )
        )
        de_s, de_note = (
            (float("nan"), "empty_context_or_answer")
            if not ctx or not ans
            else score_one_deepeval(
                q,
                ctx,
                ans,
                metric=deepeval_metric,
                model_id=mid,
                mcq_only=_default_mcq_only(),
            )
        )

        rec: Dict[str, Any] = {
            "id": mapped.get("id", ""),
            "run_folder": mapped.get("run_folder", ""),
            "source_file": mapped.get("source_file", ""),
            "benchmark": mapped.get("benchmark", ""),
            "question_id": mapped.get("question_id", ""),
            "model_name": mapped.get("model_name", ""),
            "faithfulness_prior": prior_v if prior_v == prior_v else "",
            "faithfulness_paper_hf": paper_s if paper_s == paper_s else "",
            "faithfulness_deepeval": de_s if de_s == de_s else "",
            "note_paper_hf": paper_note,
            "note_deepeval": de_note,
            "faithfulness_context_source": ctx_src,
            "faithfulness_model": mid,
            "validation_backend": "deepeval+local_hf",
        }
        results.append(rec)
        if paper_s == paper_s:
            paper_scores.append(paper_s)

        if (i + 1) % 25 == 0 or i == len(sample) - 1:
            print(
                f"  [validation {i+1}/{len(sample)}] "
                f"prior={prior_v:.0f} paper_hf={paper_s:.0f} deepeval={de_s:.0f} ctx={ctx_src}"
                if prior_v == prior_v and paper_s == paper_s and de_s == de_s
                else f"  [validation {i+1}/{len(sample)}] (partial nan)",
                flush=True,
            )

    _write_faith_batch_csv(out_path, results)

    print(f"[faithfulness] Wrote validation CSV: {out_path!r}", flush=True)

    def _paired(col_a: str, col_b: str) -> Tuple[List[float], List[float]]:
        a_out: List[float] = []
        b_out: List[float] = []
        for rec in results:
            try:
                a = float(rec.get(col_a, ""))
                b = float(rec.get(col_b, ""))
            except (TypeError, ValueError):
                continue
            if a == a and b == b:
                a_out.append(a)
                b_out.append(b)
        return a_out, b_out

    pa, pb = _paired("faithfulness_prior", "faithfulness_paper_hf")
    if pa:
        _summarize_validation_pairs("prior_stored_3b", pa, "paper_hf", pb)
    ph, pd = _paired("faithfulness_paper_hf", "faithfulness_deepeval")
    if ph:
        _summarize_validation_pairs("paper_hf", ph, "deepeval", pd)
    pr, dr = _paired("faithfulness_prior", "faithfulness_deepeval")
    if pr:
        _summarize_validation_pairs("prior_stored_3b", pr, "deepeval", dr)

    # Score distribution on validation sample (paper_hf)
    if paper_scores:
        hist = Counter(int(s // 5) * 5 for s in paper_scores)
        top = hist.most_common(3)
        print(
            "[faithfulness] Validation paper_hf top buckets: "
            + ", ".join(f"{b}-{b+4}={c}" for b, c in top),
            flush=True,
        )

    os.environ.pop("GP_FAITH_VALIDATION_DEEPEVAL", None)
    return out_path


def run_faithfulness_paper_run(
    predictions_path: str = "",
    *,
    rescore_all: bool = False,
    batch_size: int = -1,
    rag_only: Optional[bool] = None,
    norag_only: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Full or resume faithfulness run with paper settings (reretrieve + strong local judge).

    By default scores **both** RAG and NoRAG rows so Table~4 can compare grounding
    (NoRAG uses task abstract or MCQ vignette; RAG uses retrieved evidence).
    Set ``GP_FAITH_RAG_ONLY=1`` or pass ``rag_only=True`` to restrict to RAG rows only.
    Pass ``norag_only=True`` (or ``GP_FAITH_NORAG_ONLY=1``) to score only NoRAG rows
    after RAG scoring is complete.

    Outputs under ``Faithfulness/paper/``; does not consume keys from ``fathfullness/``.
    """
    apply_paper_run_env(rescore_all=rescore_all)
    acct = os.environ.get("GP_FAITH_ACCOUNT", "").strip().lower()
    if acct not in ("sifatali008", "fahim220", "fatinshadab", "ummesalmahabiba"):
        os.environ.setdefault("GP_FAITH_START_BATCH", "1")
        os.environ.setdefault("GP_FAITH_ACCOUNT", "fatinshadab")
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _KAGGLE_PREDICTIONS_CSV
    )
    norag_only = norag_only or _default_norag_only()
    if norag_only:
        rag_only = False
    elif rag_only is None:
        rag_only = os.environ.get("GP_FAITH_RAG_ONLY", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
    if rag_only and norag_only:
        raise ValueError("rag_only and norag_only are mutually exclusive")
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=rag_only,
        norag_only=norag_only,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", _PAPER_DEFAULT_MODEL),
        resume=resume,
        checkpoint_every=1,
    )


def run_faithfulness_sifatali_paper_resume(
    *,
    start_batch: int = _SIFATALI_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    predictions_path: str = "",
    batch_size: int = -1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    sifatali008 account: paper faithfulness from batch 11 onward (default).

    Example::

        run_faithfulness_sifatali_paper_resume(
            start_batch=11,
            prior_batch_dirs="/kaggle/input/myfaith/paper",
        )
    """
    apply_sifatali_paper_run_env(
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
    )
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _SIFATALI_PREDICTIONS_CSV
    )
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=True,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", _PAPER_KAGGLE_DEFAULT_MODEL),
        start_batch=start_batch,
        resume=resume,
        checkpoint_every=1,
    )


def run_faithfulness_fahim220_paper_resume(
    *,
    start_batch: int = _FAHIM220_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    predictions_path: str = "",
    batch_size: int = -1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    fahim220 account: paper faithfulness from batch 16 onward (default).

    Example::

        run_faithfulness_fahim220_paper_resume(
            start_batch=16,
            prior_batch_dirs="/kaggle/input/myfaith/paper",
        )
    """
    apply_fahim220_paper_run_env(
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
    )
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _FAHIM220_PREDICTIONS_CSV
    )
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=True,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", _PAPER_KAGGLE_DEFAULT_MODEL),
        start_batch=start_batch,
        resume=resume,
        checkpoint_every=1,
    )


def run_faithfulness_fatinshadab_paper_resume(
    *,
    start_batch: int = _FATINSHADAB_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    predictions_path: str = "",
    batch_size: int = -1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    fatinshadab account: paper faithfulness with pinned Kaggle paths.

    Example::

        run_faithfulness_fatinshadab_paper_resume(
            start_batch=11,
            prior_batch_dirs="/kaggle/input/myfaith/paper",
        )
    """
    apply_fatinshadab_paper_run_env(
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
    )
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _KAGGLE_PREDICTIONS_CSV
    )
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=True,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", _PAPER_KAGGLE_DEFAULT_MODEL),
        start_batch=start_batch,
        resume=resume,
        checkpoint_every=1,
    )


def run_faithfulness_ummesalmahabiba_paper_resume(
    *,
    start_batch: int = _UMMESALMAHABIBA_PAPER_START_BATCH,
    prior_batch_dirs: str = "",
    predictions_path: str = "",
    batch_size: int = -1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    ummesalmahabiba account: paper faithfulness with ``rag_index/`` + ``rag_index_gold/``.

    Example::

        run_faithfulness_ummesalmahabiba_paper_resume(
            start_batch=36,
            prior_batch_dirs="/kaggle/input/myfaith/paper",
        )
    """
    apply_ummesalmahabiba_paper_run_env(
        start_batch=start_batch,
        prior_batch_dirs=prior_batch_dirs,
    )
    pred = (
        predictions_path.strip()
        or os.environ.get("GP_FAITH_PREDICTIONS", "").strip()
        or _UMMESALMAHABIBA_PREDICTIONS_CSV
    )
    return run_faithfulness(
        pred,
        batch_size=batch_size,
        rag_only=True,
        backend="local_hf",
        model=os.environ.get("GP_FAITHFULNESS_MODEL", _PAPER_KAGGLE_DEFAULT_MODEL),
        start_batch=start_batch,
        resume=resume,
        checkpoint_every=1,
    )


def run_faithfulness_full_coverage(
    predictions_path: str = "",
    *,
    batch_size: int = -1,
    rag_only: bool = True,
    backend: str = "",
    model: str = "",
    resume: bool = True,
    checkpoint_every: int = 1,
) -> Dict[str, Any]:
    """
    Score remaining RAG rows until all 18,180 predictions have faithfulness (one per ``_row_key``).

    Default on Kaggle: start at **batch 31** (rows 15,001-18,180; batches 1-30 done).
    Re-run until complete, then ``build_deduped_faithfulness_final()``.
    """
    os.environ.setdefault("GP_FAITH_ROW_OFFSET", "0")
    os.environ.setdefault("GP_FAITH_LAST_ROWS", "0")
    os.environ.setdefault("GP_FAITH_START_BATCH", "31")
    os.environ.setdefault("GP_FAITH_ONLY_MISSING", "1")
    _apply_default_prior_faith_dirs()
    here = _script_dir()
    prior = os.path.join(here, "fathfullness")
    if os.path.isdir(prior):
        existing = os.environ.get("GP_FAITH_PRIOR_BATCH_DIRS", "").strip()
        os.environ["GP_FAITH_PRIOR_BATCH_DIRS"] = (
            f"{existing},{prior}" if existing else prior
        )
    return run_faithfulness(
        predictions_path,
        batch_size=batch_size,
        rag_only=rag_only,
        backend=backend,
        model=model,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    _register_pasted_notebook_module()
    if _in_jupyter_or_ipython():
        os.environ.setdefault("GP_FAITH_AUTO", "0")
        _maybe_notebook_autorun()
        if os.environ.get("GP_FAITH_AUTO", "").strip().lower() in (
            "0",
            "false",
            "no",
            "n",
            "skip",
        ):
            print(
                "[faithfulness] Paste mode ready (Cell 1 done). Next cells — no file needed:\n"
                "  run_faithfulness_paper_run()  # fatinshadab (batch 1)\n"
                "  run_faithfulness_fatinshadab_paper_resume(start_batch=11, prior_batch_dirs='...')\n"
                "  run_faithfulness_sifatali_paper_resume(start_batch=11, prior_batch_dirs='...')\n"
                "  run_faithfulness_fahim220_paper_resume(start_batch=16, prior_batch_dirs='...')\n"
                "  run_faithfulness_ummesalmahabiba_paper_resume(start_batch=36, prior_batch_dirs='...')\n"
                "  run_faithfulness_ummesalmahabiba_gap_fill()  # last 2500 missing rows\n"
                "  run_faithfulness_fahim220_gap_fill()  # fahim220 gap-fill paths\n"
                "  apply_gap_fill_run_env(...); run_faithfulness(...)\n"
                "  build_deduped_faithfulness_final('/kaggle/working/Faithfulness/paper')",
                flush=True,
            )
    else:
        main()