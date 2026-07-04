"""
Open-source HF benchmarks only (no local dataset files required).

**Single-file Kaggle:** Paste or upload only ``eval_benchmarks.py``. On first import it
base64-decodes and extracts ``real_model_runner.py``, ``eval_quality_metrics.py``, and
``measurement_config.py`` into ``/kaggle/working`` (or ``.`` locally if helpers are missing)
when that folder does not already contain all three files—delete those files there if you need
to refresh helpers after updating this script.
Install pip deps, set ``HF_TOKEN`` for gated Hub models and dataset access; no companion ``.py`` files or zip dataset are required.
For local runs with **only** public Hub models/datasets, you may set ``GP_BENCH_ALLOW_NO_HF_TOKEN=1`` to skip the token check (not for Kaggle Secrets workflows).

**Local GPU models:** unpacked ``real_model_runner.py`` defaults to **Gemma-2-2b-it** (SLM) and **Llama-2-7b-chat** (LLM). Set ``HF_TOKEN`` for gated Hub access. Override with ``GP_MODEL_SLM`` / ``GP_MODEL_LLM`` before starting the kernel if needed.

**Kaggle setup (run in order):**

1. **Cell 1 — absolute first cell** (set env vars before any ``datasets`` import, then pandas)::

       import os
       os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
       os.environ["TQDM_DISABLE"] = "1"
       !pip install -q 'numpy>=1.26.4,<2.1' 'scipy>=1.11.4,<1.15' \\
        'pandas>=2.0,<3.0' 'datasets>=2.14' pyarrow --force-reinstall
       # Kernel → Restart Session

   **Cell 1b — Hugging Face stack** (after restart; do **not** ``--force-reinstall`` torch)::

       !pip install -q 'transformers>=4.43.0,<5' 'huggingface-hub>=0.23.0' \\
        'accelerate>=0.26.0' 'sentence-transformers>=2.2.2' 'safetensors>=0.4.0' \\
        faiss-cpu 'bitsandbytes>=0.46.1' rouge-score sacrebleu nltk bert-score openai --upgrade
       # Kernel → Restart Session again (required before Gemma-2 / RAG embed load)

2. **Cell 2** — verify (FAISS required for dense RAG / differentiated recall@k)::

       import faiss, numpy, scipy, pandas as pd, transformers
       from transformers import PreTrainedModel, AutoModelForCausalLM
       print(faiss.__version__, numpy.__version__, scipy.__version__, pd.__version__, transformers.__version__)

3. **Cell 3** — paste the **whole** ``eval_benchmarks.py`` in **one cell** (do **not** paste only
   the last ``if __name__ == "__main__"`` block; that causes ``NameError: os is not defined``).
   Optional CLI args at the top of the same cell::

       import sys
       sys.argv = ["eval_benchmarks.py", "--benchmark", "all", "--max_items", "100", "--seed", "42"]
       # paste the FULL eval_benchmarks.py below

   On first run the script auto-saves to ``/kaggle/working/eval_benchmarks.py`` so later cells can use::

       !python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 100 --seed 42

   RAG uses ``/kaggle/input/datasets/salmashopna/rag-index/`` when that dataset is attached
   (``chunks.jsonl``, ``index.faiss``, paper-556 sidecars; fallbacks: ``hafijur222/rag-index``,
   ``sifatali008/rag-index``, etc.).
   PubMedQA recall@k gold index: attach ``salmashopna/rag-index-gold`` (``pubmedqa_gold_manifest.json``)
   or build
   ``/kaggle/working/rag_index_gold``; ``--auto_gold_pubmedqa`` discovers the Kaggle dataset automatically.
   Set ``GP_BENCH_STRICT=1`` to hard-fail on missing faiss. Set ``GP_BENCH_ALLOW_NOTEBOOK_RUN=1`` to
   disable auto re-exec (not recommended).

   In-process ``%%run eval_benchmarks.py`` breaks ``transformers`` after Cell 1b pip; always spawn
   a fresh interpreter with ``!python``.

``ChunkLoadError: HBoxModel`` / ``Failed to load HBoxModel`` is harmless **browser console** noise
from Jupyter widgets when HF tries to show a progress bar. Python keeps running; benchmarks are
unaffected. Fix: run Cell 1 env vars **before** any ``import datasets``, restart kernel, then run
this script. The script calls ``_quiet_hf_datasets()`` at startup on Kaggle.

**bitsandbytes:** If missing, use Cell 1 for ``pip install -q -U 'bitsandbytes>=0.46.1'`` (+ ``datasets``,
``rouge-score``, ``sacrebleu``, ``nltk``, ``bert-score``, ``accelerate``, ``faiss-cpu``,
``sentence-transformers``), then restart before
``torch``/``transformers``. Never ``pip install pandas>=2.0`` without ``<3.0`` (breaks Kaggle).
Do not keep ``pandas.py`` or ``datasets.py`` in ``/kaggle/working``. Runtime HF auto-pip is off by default
(``GP_BENCH_ALLOW_AUTO_PIP=1`` to opt in; never re-import in the same kernel after pip).

- **MedQA (USMLE-style):** ``nnilayy/medqa-usmle`` (validation).
- **MMLU-Med:** ``cais/mmlu`` — six medical subjects; default ``dev`` split (``--mmlu_split test`` for
  legacy comparison; ``test`` may overlap pretraining—disclose in paper).
- **PubMedQA:** ``pubmed_qa`` / ``pqa_labeled`` — HF only exposes ``train``; eval uses a **fixed holdout**
  (default 20%, seed 42) so gold-index chunks exclude eval PMIDs. Label accuracy on holdout only.
  Generative BLEU/ROUGE/BERTScore are **disabled** for one-word yes/no/maybe outputs.
  **Do not use ``--mock``** when PubMedQA is in the run (real free-text answers only).
- **MedMCQA (optional):** ``medmcqa`` — use ``--benchmark medmcqa``; not in default ``all``.

Default ``--benchmark all`` runs **MedQA + MMLU-Med + PubMedQA** (pick **either** MedMCQA **or**
MMLU-Med as your second MCQ track; here MMLU-Med is the default pair with MedQA).

**Step 1:** Subset with ``--max_items`` (default **100**; use **500** for full paper runs; **10** for quick smoke). Each run draws a random N per benchmark
unless you pass ``--seed 42`` to fix the draw (logged at startup). Use ``--first_n`` for the first N in
dataset order. Per-row predictions are **saved by
default** next to ``--out_json`` as ``<stem>_predictions.json`` (e.g. ``benchmark_results_all_predictions.json``).
Override with ``--save_predictions /prefix`` or skip with ``--no_save_predictions``. When the benchmark
supplies it (PubMedQA), each row includes full ``context`` (abstract) for RAG ``chunks.jsonl`` export.

**Output:** After a successful run, aggregated metrics are written to ``benchmark_results_all.json``
next to this script when running locally (avoids mixing up same-named files in other folders), or
``/kaggle/working/benchmark_results_all.json`` on Kaggle. Use ``--out_json /path/to/file.json`` to
choose a different path.

Four configs: SLM_NoRAG, SLM_RAG, LLM_NoRAG, LLM_RAG. Each SLM/LLM loads **once** for all
benchmarks (not per dataset). RAG FAISS + hybrid reranker are **prewarmed**; cross-encoder is
**cached** (no reload per question).

**RAG index paths (used in place when found):**

- **Google Colab + Drive:** ``/content/drive/MyDrive/Colab Notebooks/rag_index/`` (``chunks.jsonl`` + ``index.faiss``)
- **Kaggle:** ``/kaggle/input/datasets/salmashopna/rag-index/`` — ``chunks.jsonl``, ``index.faiss``,
  ``chunks_paper_556.jsonl``, ``index_paper_556.faiss``, ``paper_corpus_stats.json``,
  ``paper_source_manifest.jsonl`` (same folder). Gold: ``salmashopna/rag-index-gold`` +
  ``pubmedqa_gold_manifest.json``. Fallbacks: ``hafijur222/rag-index``, ``sifatali008/rag-index``.

**Local / repo ``kaggle_working/rag_index`` (two corpora):**

- **Default (paper experiments & benchmarks):** ``chunks.jsonl`` + ``index.faiss`` — typically **~25k** non-leaking
  chunks (``build_external_corpus.py`` + ``build_rag_index.py``).
- **IEEE Access 115-doc corpus:** ``chunks_paper_556.jsonl`` + ``index_paper_556.faiss`` (556 segments);
  optional metadata ``paper_corpus_stats.json``, ``paper_source_manifest.jsonl``.
  Use ``--rag_paper_corpus`` or ``GP_RAG_PAPER_556=1`` with ``--rag_index_dir`` pointing at that folder
  (or rely on ``/kaggle/working/rag_index`` after you copy both file pairs there).

Override with ``GP_RAG_DATASET_DIR`` or ``--rag_index_dir``. Colab staging copy (optional):
``/content/rag_index``. Kaggle staging: ``/kaggle/working/rag_index`` (set ``GP_RAG_FORCE_STAGE=1``).

Build locally: ``python build_rag_index.py --input chunks.jsonl --out_dir /path``.
Paper corpus: ``python build_paper_corpus.py --out .../chunks_paper_556.jsonl`` then
``python build_rag_index.py --input .../chunks_paper_556.jsonl --out_dir ...`` and rename outputs
to ``index_paper_556.faiss`` / ``chunks_paper_556.jsonl`` if you keep both trees in one folder.
Multi-window corpus (differentiated recall@k): ``python eval_benchmarks.py --build_fixed_chunks path/to/chunks.jsonl``.
``--rag_top_k`` (default 3), ``--rag_context_max_chars`` (2000), hybrid RRF+BM25+rerank (``rag_retrieval.py``).
Reranker ``UNEXPECTED position_ids`` load notes are harmless. Set ``RAG_QUIET=1`` to reduce per-query logs;
``RAG_VERBOSE=1`` logs every hybrid timing line.
``--no_rag_stage`` to skip
auto-copy from input. Seed auto-build runs only when no real index is found (``RAG_AUTO_BUILD=0`` to disable).
On Kaggle, ``bitsandbytes`` may be **one-shot pip-installed** when absent (see bitsandbytes note above);
faiss/sentence-transformers still follow the same Cell 1 + restart guidance if skipped at runtime.
Optional env: ``GP_BENCH_NO_AUTO_PIP=1`` blocks any auto-pip; ``GP_BENCH_KAGGLE_AUTO_PIP=1`` forces legacy
in-process upgrades on Kaggle only.

**Hugging Face Inference Router (no local GPU):** pass ``--hf_router`` to call the OpenAI-compatible
API at ``https://router.huggingface.co/v1`` using ``HF_TOKEN`` as the API key (``pip install openai``).
Override models with ``--hf_router_model_slm`` / ``--hf_router_model_llm`` or env ``HF_ROUTER_MODEL_SLM`` /
``HF_ROUTER_MODEL_LLM`` (include provider suffix when required, e.g. ``meta-llama/Llama-2-7b-chat-hf`` or ``...:featherless-ai`` on the router).
RAG still uses ``real_model_runner.build_rag_context`` (FAISS / seed index) after helpers are unpacked.

**Token F1 (important):** ``token_f1`` uses real token overlap via ``eval_quality_metrics``. For MCQ,
reference text is the **full option string** (not the letter A–D). For PubMedQA, reference text is
``long_answer`` (not the yes/no/maybe label); ``token_f1_label`` is the degenerate label-only F1 for
comparison. ROUGE/BLEU/METEOR/BERTScore on PubMedQA also use ``long_answer``.

**Retrieval metrics (PubMedQA + PubMed corpus):** When the RAG index includes ``pubmedqa_<pmid>``
chunks (legacy / PubMedQA-aligned corpus), RAG configs log per-row ``recall_at_{1,3,5,10}`` and ``mrr``
(gold ``pubmedqa_<pubid>`` vs ``rag_ranked_sources``). With the default **~25k external** corpus
(MedQuad + PubMed abstracts, no ``pubmedqa_*`` sources), those metrics are **not interpretable**
(treat retrieval eval as N/A or use a gold-aligned index). Summarize after a run with
``summarize_rag_in_predictions(...)``. Tune K list via env ``RAG_RECALL_K=1,5,10``; ranking depth via ``RAG_METRICS_MAX_K`` (default 50).

**Gold-aligned PubMedQA index (recall@k fix):** external 25k has no ``pubmedqa_*`` chunks — build gold once, then either swap index or use dual corpus::

    python eval_benchmarks.py --build_gold_index /kaggle/working/rag_index_gold
    # (writes chunks.jsonl + index.faiss in one step)

    # Dual corpus (recommended): external index for MCQ; gold holdout index for PubMedQA RAG + recall@k
    python eval_benchmarks.py --build_gold_index /kaggle/working/rag_index_gold
    python eval_benchmarks.py --benchmark all --auto_gold_pubmedqa --seed 42

Gold index is built from **corpus holdout only** (eval PMIDs excluded). Retrieval queries use **question only**
(set ``GP_PUBMEDQA_RETRIEVAL_QUERY_INCLUDES_ABSTRACT=1`` only for ablation). Report MCQ RAG and PubMedQA RAG in
**separate tables** (``rag_experiment`` in results JSON).

Or: ``python build_pubmedqa_gold_corpus.py`` (overlapping word windows). Set ``RAG_MIN_SCORE=0.25`` (default)
to skip low-confidence retrieval injection.

Use the 25k external index for non-leaking MCQ RAG task comparisons. MedQA/MMLU have no chunk gold IDs.

**Faculty-review post-hoc (Tables X–Z, H, S1–S2):** after benchmarks, run::

    python run_paper_eval_suite.py \\
        --benchmark_json benchmark_results_all.json \\
        --predictions_json benchmark_results_all_predictions.json \\
        --run_judge --run_faithfulness

Or chain in one command: ``python eval_benchmarks.py --benchmark all --post_eval``.
Faithfulness: ``run_faithfulness_eval.py`` (DeepEval + local HF on Kaggle; or ``--backend openrouter``).
LLM judge: ``llm_judge.py``. Tables: ``eval_posthoc.py`` / ``paper_tables.json``.

    pip install -q "pandas>=2.0,<3.0" "datasets>=2.14" pyarrow rouge-score sacrebleu nltk bert-score accelerate openai
    python eval_benchmarks.py --max_items 500
    python eval_benchmarks.py --max_items 500 --seed 42   # same questions every time
    python eval_benchmarks.py --benchmark all --max_items 500 --post_eval --post_eval_faithfulness_limit 200
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import secrets
import shutil
import statistics
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Before any HF datasets import (reduces HBoxModel ChunkLoadError in Kaggle Jupyter).
# Use assignment (not setdefault): notebook cells may have set empty values earlier.
if os.path.isdir("/kaggle"):
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"

import base64
import io
import zipfile

# Bump when helper sources change; stale /kaggle/working copies are refreshed on mismatch.
GP_BUNDLE_VERSION = "20260527d"

# Single-file bundle: measurement_config, eval_quality_metrics, real_model_runner (extract on demand).
_BUNDLED_PY_ZIP_B64 = """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"""


def _bundle_stamp_path(work_dir: str) -> str:
    return os.path.join(work_dir, ".gp_bench_bundle_version")


def _read_bundle_stamp(work_dir: str) -> str:
    path = _bundle_stamp_path(work_dir)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_bundle_stamp(work_dir: str) -> None:
    with open(_bundle_stamp_path(work_dir), "w", encoding="utf-8") as f:
        f.write(GP_BUNDLE_VERSION)


def _remove_helper_bundle(work_dir: str) -> None:
    for name in _HELPER_MODULES:
        path = os.path.join(work_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _sync_kaggle_helper_bundle(work_dir: str) -> None:
    """
    Replace stale ``/kaggle/working`` helpers from an older notebook run.

    A complete three-file tree with an old ``GP_BUNDLE_VERSION`` otherwise blocks unpack.
    """
    stamp = _read_bundle_stamp(work_dir)
    if stamp == GP_BUNDLE_VERSION and _dir_has_complete_helper_bundle(work_dir):
        return
    if _dir_has_complete_helper_bundle(work_dir) or _dir_has_runner(work_dir):
        print(
            f"GP_BENCH: refreshing helper bundle in {work_dir} "
            f"(was {stamp or 'unversioned'}, need {GP_BUNDLE_VERSION}).",
            flush=True,
        )
        _remove_helper_bundle(work_dir)
    _unpack_bundled_helpers(work_dir)
    _write_bundle_stamp(work_dir)


def _unpack_bundled_helpers(work_dir: str) -> None:
    raw = base64.b64decode(_BUNDLED_PY_ZIP_B64)
    os.makedirs(work_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        zf.extractall(work_dir)
    print(f"Unpacked bundled helpers into {work_dir}", flush=True)



def _on_colab() -> bool:
    return os.path.isdir("/content") and not os.path.isdir("/kaggle")


def _work_dir() -> str:
    if os.path.isdir("/kaggle"):
        return "/kaggle/working"
    if _on_colab():
        return "/content"
    return "."


def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return ""


def _default_out_json() -> str:
    # Kaggle: fixed working path. Local: same folder as this script so cwd/other trees
    # do not collide with another benchmark_results_all.json.
    if os.path.isdir("/kaggle/working"):
        return os.path.join("/kaggle/working", "benchmark_results_all.json")
    sd = _script_dir()
    if sd:
        return os.path.join(sd, "benchmark_results_all.json")
    return os.path.join(os.getcwd(), "benchmark_results_all.json")


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


_HELPER_MODULES = (
    "real_model_runner.py",
    "eval_quality_metrics.py",
    "measurement_config.py",
)


def _dir_has_complete_helper_bundle(root: str) -> bool:
    return all(os.path.isfile(os.path.join(root, n)) for n in _HELPER_MODULES)


def _dir_has_runner(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "real_model_runner.py"))


def _code_roots_for_import() -> List[str]:
    """
    Search order for real_model_runner (and peers). Prefer a *complete* three-file bundle
    so a stray real_model_runner.py alone under /kaggle/input does not win over this script's folder.
    """
    roots: List[str] = []
    cd = (os.environ.get("CODE_DIR") or os.environ.get("GREEN_PAPER_CODE") or "").strip()
    if cd:
        roots.append(cd)
    sd = _script_dir()
    if sd:
        roots.append(sd)
    roots.append(_work_dir())
    roots.extend(_kaggle_input_code_dirs())
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
        roots = _code_roots_for_import()
        for prefer_complete in (True, False):
            for root in roots:
                ok = _dir_has_complete_helper_bundle(root) if prefer_complete else _dir_has_runner(root)
                if not ok:
                    continue
                if root not in sys.path:
                    sys.path.insert(0, root)
                return root
        return None

    if on_kaggle:
        _sync_kaggle_helper_bundle(work)
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
    """Reload helper modules only; do not clear FAISS/reranker/BM25 caches mid-run."""
    for mod in ("real_model_runner", "eval_quality_metrics", "measurement_config"):
        sys.modules.pop(mod, None)


def _inference_telemetry(raw_or_obj: Any) -> Dict[str, Any]:
    """Extract per-query latency / NVML energy fields from run_fn dict returns."""
    if not isinstance(raw_or_obj, dict):
        return {}
    out: Dict[str, Any] = {}
    lat = raw_or_obj.get("latency_seconds")
    if lat in (None, "") and raw_or_obj.get("latency") not in (None, ""):
        lat = raw_or_obj.get("latency")
    if lat not in (None, ""):
        try:
            out["latency_seconds"] = round(float(lat), 6)
        except (TypeError, ValueError):
            pass
    for key in (
        "response_tokens",
        "energy_joules_nvml",
        "gpu_energy_kwh_nvml",
        "gpu_power_watts_mean_nvml",
        "gpu_name",
        "quantization_backend",
    ):
        val = raw_or_obj.get(key)
        if val not in (None, ""):
            out[key] = val
    return out


def _run_out(
    raw_or_obj: Any,
) -> Tuple[str, str, str, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Normalize run_fn return: response, context, source, hits, ranked_sources, rag_diagnostic."""
    if isinstance(raw_or_obj, dict):
        hits = raw_or_obj.get("rag_hits")
        if not isinstance(hits, list):
            hits = []
        ranked = raw_or_obj.get("rag_ranked_sources")
        if not isinstance(ranked, list):
            ranked = []
        diag = raw_or_obj.get("rag_diagnostic")
        if not isinstance(diag, dict):
            diag = {}
        return (
            str(raw_or_obj.get("response", "")),
            str(raw_or_obj.get("retrieved_context") or raw_or_obj.get("evidence") or ""),
            str(raw_or_obj.get("rag_source") or ""),
            hits,
            [str(s) for s in ranked if str(s)],
            diag,
        )
    return str(raw_or_obj), "", "", [], [], {}


def _item_iterator(items: List[Dict[str, Any]], desc: str = "") -> Any:
    """Optional tqdm progress when GP_BENCH_TQDM=1."""
    if _env_truthy("GP_BENCH_TQDM"):
        try:
            from tqdm import tqdm

            return tqdm(items, desc=desc, unit="q")
        except ImportError:
            pass
    return items


def _rag_min_score() -> float:
    try:
        return float(os.environ.get("RAG_MIN_SCORE", "0.25"))
    except ValueError:
        return 0.25


def _hit_retrieval_score(hit: Dict[str, Any]) -> float:
    if not isinstance(hit, dict):
        return 0.0
    for key in ("reranker_score", "dense_score", "score", "similarity", "combined_score"):
        v = hit.get(key)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _filter_rag_hits_by_score(
    hits: List[Dict[str, Any]],
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Drop low-confidence hits so weak retrieval is not injected as context."""
    if not hits:
        return hits
    thr = _rag_min_score() if min_score is None else min_score
    filtered = [h for h in hits if isinstance(h, dict) and _hit_retrieval_score(h) >= thr]
    return filtered


def _should_use_rag_context(
    hits: List[Dict[str, Any]],
    retrieved_context: str,
    min_score: Optional[float] = None,
    min_context_chars: int = 50,
) -> bool:
    """Skip RAG injection when retrieval quality is too low (reduces harm vs NoRAG)."""
    if not retrieved_context or len(retrieved_context.strip()) < min_context_chars:
        return False
    if not hits:
        return False
    thr = _rag_min_score() if min_score is None else min_score
    good = _filter_rag_hits_by_score(hits, thr)
    return bool(good)


def _prefetch_rag_for_query(query: str) -> Tuple[str, List[Dict[str, Any]], List[str], str, Dict[str, Any]]:
    """Retrieve context without running generation (for gated prompt assembly)."""
    try:
        _setup_runner_path()
        import real_model_runner as rmr

        block, _ev, src = rmr.build_rag_context(query, True)
        diag = getattr(rmr, "LAST_RAG_DIAGNOSTIC", None) or getattr(
            rmr, "LAST_RETRIEVAL_DIAGNOSTIC", None
        )
        ranked = [str(s) for s in (rmr.LAST_RAG_RANKED_SOURCES or []) if str(s)]
        hits = list(rmr.LAST_RAG_HITS or [])
        diag_out = dict(diag) if isinstance(diag, dict) else {}
        if not diag_out.get("retrieved_chunk_ids"):
            diag_out["retrieved_chunk_ids"] = _chunk_ids_from_rag(hits, ranked, diag_out)
        ranked = _enrich_ranked_sources_from_diag(ranked, diag_out)
        return (
            str(block or ""),
            hits,
            ranked,
            str(src or ""),
            diag_out,
        )
    except Exception as ex:
        return "", [], [], "prefetch_failed", {"error": str(ex)}


def _clear_rag_runtime_cache() -> None:
    _setup_runner_path()
    import real_model_runner as rmr

    rmr._clear_rag_cache()


@contextmanager
def _temporary_rag_corpus(index_dir: str):
    """Pin a different FAISS/chunks dir for one retrieval call (PubMedQA gold index)."""
    saved = _rag_env_snapshot()
    try:
        _pin_rag_environment(index_dir)
        _clear_rag_runtime_cache()
        yield
    finally:
        for key in saved:
            val = saved[key]
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
        _clear_rag_runtime_cache()


def _default_gold_rag_build_dir() -> str:
    """Writable target for ``--build_gold_index`` (not necessarily where a prebuilt index lives)."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/rag_index_gold"
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "kaggle_working", "rag_index_gold")


def _gold_rag_dir_candidates() -> List[str]:
    """Search order for a ready PubMedQA gold index (``pubmedqa_*`` in ``chunks.jsonl``)."""
    dirs: List[str] = []
    explicit = (os.environ.get("GP_RAG_GOLD_INDEX_DIR") or "").strip()
    if explicit:
        dirs.append(explicit)
    custom_ds = (os.environ.get("GP_RAG_GOLD_DATASET_DIR") or "").strip()
    if custom_ds:
        dirs.append(custom_ds)
    if os.path.isdir("/kaggle"):
        dirs.extend(_discover_gold_rag_dirs())
        dirs.extend(
            [
                "/kaggle/working/rag_index_gold",
                KAGGLE_GOLD_RAG_DATASET_DIR,
                "/kaggle/input/salmashopna/rag-index-gold",
                "/kaggle/input/datasets/salmashopna/rag-index-gold",
                "/kaggle/input/hafijur222/rag-index-gold",
                "/kaggle/input/datasets/hafijur222/rag-index-gold",
            ]
        )
        for base in (
            KAGGLE_GOLD_RAG_DATASET_DIR,
            "/kaggle/input/salmashopna/rag-index-gold",
            "/kaggle/input/datasets/salmashopna/rag-index-gold",
            "/kaggle/input/hafijur222/rag-index-gold",
            "/kaggle/input/datasets/hafijur222/rag-index-gold",
        ):
            for sub in (
                "",
                "rag_index_gold",
                os.path.join("datasets", "salmashopna", "rag-index-gold"),
                os.path.join("datasets", "hafijur222", "rag-index-gold"),
            ):
                if sub:
                    dirs.append(os.path.join(base, sub))
    sd = _script_dir()
    if sd:
        dirs.append(os.path.join(sd, "kaggle_working", "rag_index_gold"))
    dirs.append(_default_gold_rag_build_dir())
    out: List[str] = []
    seen: Set[str] = set()
    for d in dirs:
        if not d:
            continue
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            out.append(ad)
    return out


def _discover_gold_rag_dirs() -> List[str]:
    """Walk ``/kaggle/input`` for any folder with gold ``pubmedqa_*`` chunks + FAISS index."""
    found: List[str] = []
    seen: Set[str] = set()
    for d in (
        KAGGLE_GOLD_RAG_DATASET_DIR,
        "/kaggle/input/salmashopna/rag-index-gold",
        "/kaggle/input/datasets/salmashopna/rag-index-gold",
        "/kaggle/input/hafijur222/rag-index-gold",
        "/kaggle/input/datasets/hafijur222/rag-index-gold",
        "/kaggle/working/rag_index_gold",
    ):
        if _gold_rag_index_ready(d):
            ad = os.path.abspath(d)
            if ad not in seen:
                seen.add(ad)
                found.append(ad)
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            root = os.path.join(kin, name)
            if _gold_rag_index_ready(root):
                ad = os.path.abspath(root)
                if ad not in seen:
                    seen.add(ad)
                    found.append(ad)
            for d in _walk_rag_index_dirs(root, max_depth=5):
                if _gold_rag_index_ready(d):
                    ad = os.path.abspath(d)
                    if ad not in seen:
                        seen.add(ad)
                        found.append(ad)
    return found


def _print_gold_index_probe() -> None:
    print("--- PubMedQA gold index probe (first paths checked) ---", flush=True)
    for p in _gold_rag_dir_candidates()[:14]:
        cp = os.path.join(p, "chunks.jsonl")
        ip = os.path.join(p, "index.faiss")
        pub = 0
        if os.path.isfile(cp):
            _n, _seed, pub = _rag_chunks_metadata(cp)
        ready = _gold_rag_index_ready(p)
        manifest = os.path.join(p, "pubmedqa_gold_manifest.json")
        print(
            f"  {p!r}\n"
            f"    ready={ready} chunks={os.path.isfile(cp)} faiss={os.path.isfile(ip)} "
            f"pubmedqa_hits={pub} manifest={os.path.isfile(manifest)}",
            flush=True,
        )


def _pin_pubmedqa_gold_for_run(benchmarks: List[str]) -> str:
    """Pin ``GP_RAG_GOLD_INDEX_DIR`` when PubMedQA is in the run (no extra CLI flag required)."""
    if "pubmedqa" not in benchmarks:
        return ""
    if _env_truthy("GP_DISABLE_AUTO_GOLD_PUBMEDQA"):
        return os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip()
    cur = os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip()
    if cur and _gold_rag_index_ready(cur):
        return cur
    found = _resolve_gold_rag_dir()
    if found:
        os.environ["GP_RAG_GOLD_INDEX_DIR"] = found
        gcp = os.path.join(found, "chunks.jsonl")
        n_total, _is_seed, pubmed_hits = _rag_chunks_metadata(gcp)
        print(
            f"\n{'=' * 60}\n"
            f"PUBMEDQA GOLD INDEX: {found!r}\n"
            f"  {pubmed_hits} pubmedqa_* / {n_total} chunks — dual corpus active for recall@k\n"
            f"{'=' * 60}\n",
            flush=True,
        )
        return found
    print(
        "\n*** PubMedQA gold index NOT FOUND — recall@k will stay 0 / N/A ***\n"
        "  Attach Kaggle dataset: salmashopna/rag-index-gold\n"
        f"  Or: os.environ['GP_RAG_GOLD_INDEX_DIR'] = {KAGGLE_GOLD_RAG_DATASET_DIR!r}\n",
        flush=True,
    )
    _print_gold_index_probe()
    return ""


def _resolve_gold_rag_dir() -> str:
    """First gold index dir with ``chunks.jsonl`` + ``index.faiss`` and ``pubmedqa_*`` rows."""
    for d in _gold_rag_dir_candidates():
        if _gold_rag_index_ready(d):
            return d
    return ""


def _default_gold_rag_dir() -> str:
    """Best gold path for messages; prefers a discovered ready index over the build default."""
    found = _resolve_gold_rag_dir()
    if found:
        return found
    return _default_gold_rag_build_dir()


def _gold_rag_index_ready(index_dir: str) -> bool:
    d = os.path.abspath(index_dir.strip())
    cp = os.path.join(d, "chunks.jsonl")
    ip = os.path.join(d, "index.faiss")
    if not (os.path.isfile(cp) and os.path.isfile(ip)):
        return False
    _n, _is_seed, pubmed_hits = _rag_chunks_metadata(cp)
    return pubmed_hits > 0


def _pubmedqa_rag_index_dir() -> str:
    return os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip()


def _pubmedqa_retrieval_index_dir() -> str:
    """Directory with ``pubmedqa_*`` chunks for PubMedQA recall@k (gold or primary)."""
    gold = _pubmedqa_rag_index_dir()
    if gold and _gold_rag_index_ready(gold):
        return gold
    primary = os.environ.get("RAG_INDEX_DIR", "").strip()
    if primary and _gold_rag_index_ready(primary):
        return primary
    return ""


def _pubmedqa_retrieval_query_includes_abstract() -> bool:
    """Default False: including the eval abstract in the retrieval query inflates Recall@K (audit C2)."""
    return _env_truthy("GP_PUBMEDQA_RETRIEVAL_QUERY_INCLUDES_ABSTRACT")


def _pubmedqa_retrieval_query(item: Dict[str, Any]) -> str:
    """
    Query text for PubMedQA FAISS/BM25 (not the generation prompt).

    Default: **question only**. Generation still receives the full abstract in the user prompt.
    """
    q = str(item.get("question") or "").strip()
    if not _pubmedqa_retrieval_query_includes_abstract():
        return q
    ctx = str(item.get("context") or "").strip()
    if not ctx:
        return q
    try:
        cap = int(os.environ.get("PUBMEDQA_RETRIEVAL_CONTEXT_CHARS", "4000"))
    except ValueError:
        cap = 4000
    cap = max(500, min(12000, cap))
    return f"{q}\n\n{ctx[:cap]}" if q else ctx[:cap]


@contextmanager
def _pubmedqa_gold_retrieval_tuning():
    """Relax dense gate on small gold corpora so the labeled chunk stays in the metrics list."""
    saved: Dict[str, str] = {}
    if not _pubmedqa_retrieval_index_dir():
        yield
        return
    for key, default in (
        ("RAG_MIN_DENSE_SCORE", "0.12"),
        ("RAG_FETCH_MULT", "8"),
        ("RAG_RERANK_CANDIDATES", "30"),
    ):
        env_key = f"GP_GOLD_{key}"
        new_val = os.environ.get(env_key, default).strip() or default
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


def _enrich_ranked_sources_from_diag(
    ranked_sources: List[str],
    rag_diagnostic: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Prefer full ``retrieved_chunk_ids`` from hybrid (up to RAG_METRICS_MAX_K) over prompt top-k."""
    diag = rag_diagnostic or {}
    extra: List[str] = []
    for key in ("retrieved_chunk_ids", "final_chunk_ids"):
        raw = diag.get(key)
        if isinstance(raw, list):
            extra = [str(x) for x in raw if str(x)]
        elif isinstance(raw, str) and raw.strip().startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    extra = [str(x) for x in parsed if str(x)]
            except json.JSONDecodeError:
                pass
        if extra:
            break
    if len(extra) > len(ranked_sources):
        return extra
    return list(ranked_sources)


def _primary_rag_index_dir() -> str:
    return os.environ.get("RAG_INDEX_DIR", "").strip()


def _prefetch_rag_for_pubmedqa_generation(
    item: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[str], str, Dict[str, Any]]:
    """PubMedQA **task** RAG: always primary external index (no eval-abstract oracle)."""
    query = _pubmedqa_retrieval_query(item)
    primary = _primary_rag_index_dir()
    if primary and os.path.isfile(os.path.join(primary, "chunks.jsonl")):
        with _temporary_rag_corpus(primary):
            block, hits, ranked, src, diag = _prefetch_rag_for_query(query)
    else:
        block, hits, ranked, src, diag = _prefetch_rag_for_query(query)
    ranked = _enrich_ranked_sources_from_diag(ranked, diag)
    if isinstance(diag, dict):
        diag = dict(diag)
        diag["pubmedqa_rag_role"] = "task_generation"
        diag["generation_rag_index_dir"] = primary or None
        diag["retrieval_query_mode"] = (
            "question_plus_abstract"
            if _pubmedqa_retrieval_query_includes_abstract()
            else "question_only"
        )
        diag["retrieval_query"] = query[:500]
    return block, hits, ranked, src, diag


def _prefetch_rag_for_pubmedqa_retrieval_eval(
    item: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[str], str, Dict[str, Any]]:
    """PubMedQA **recall@k** prefetch on gold index (must include eval-holdout ``pubmedqa_<pmid>`` chunks)."""
    query = _pubmedqa_retrieval_query(item)
    idx = _pubmedqa_retrieval_index_dir()
    if idx:
        with _temporary_rag_corpus(idx), _pubmedqa_gold_retrieval_tuning():
            block, hits, ranked, src, diag = _prefetch_rag_for_query(query)
    else:
        return "", [], [], "none", {"retrieval_eval_index_dir": None}
    ranked = _enrich_ranked_sources_from_diag(ranked, diag)
    if isinstance(diag, dict):
        diag = dict(diag)
        diag["pubmedqa_rag_role"] = "retrieval_eval"
        diag["retrieval_eval_index_dir"] = idx
        diag["retrieval_query_mode"] = (
            "question_plus_abstract"
            if _pubmedqa_retrieval_query_includes_abstract()
            else "question_only"
        )
        diag["retrieval_query"] = query[:500]
    return block, hits, ranked, src, diag


def _prefetch_rag_for_pubmedqa(
    item: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[str], str, Dict[str, Any]]:
    """
    Dual prefetch (C4): external index for generation context; gold index for ranked ids / recall@k.
    """
    gen_block, gen_hits, _gen_ranked, gen_src, gen_diag = _prefetch_rag_for_pubmedqa_generation(item)
    _met_block, _met_hits, met_ranked, met_src, met_diag = _prefetch_rag_for_pubmedqa_retrieval_eval(
        item
    )
    diag: Dict[str, Any] = dict(met_diag or {})
    if gen_diag:
        diag["generation_rag_index_dir"] = gen_diag.get("generation_rag_index_dir")
        diag["generation_rag_source"] = gen_src
    diag["retrieval_eval_index_dir"] = met_diag.get("retrieval_eval_index_dir")
    return gen_block, gen_hits, met_ranked, gen_src, diag


def _pubmedqa_gold_rag_active() -> bool:
    return bool(_pubmedqa_retrieval_index_dir())


def _auto_enable_gold_pubmedqa_index_if_needed(benchmarks: List[str]) -> Dict[str, Any]:
    """
    When the primary corpus has no ``pubmedqa_*`` rows but a gold index exists on disk,
    pin ``GP_RAG_GOLD_INDEX_DIR`` so recall@k / MRR are computable without extra CLI flags.
    """
    if "pubmedqa" not in benchmarks:
        return {"auto_gold_enabled": False, "reason": "pubmedqa_not_in_benchmarks"}
    if _env_truthy("GP_DISABLE_AUTO_GOLD_PUBMEDQA"):
        return {
            "auto_gold_enabled": False,
            "reason": "GP_DISABLE_AUTO_GOLD_PUBMEDQA=1",
            "paper_note": "Set --auto_gold_pubmedqa or unset GP_DISABLE_AUTO_GOLD_PUBMEDQA for paper runs.",
        }
    if _pubmedqa_gold_rag_active():
        return {
            "auto_gold_enabled": False,
            "reason": "gold_index_already_active",
            "gold_dir": os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip() or None,
        }
    if _rag_corpus_pubmed_hits() > 0:
        return {
            "auto_gold_enabled": False,
            "reason": "primary_corpus_has_pubmedqa_chunks",
        }
    cand = _resolve_gold_rag_dir()
    if not cand:
        return {
            "auto_gold_enabled": False,
            "reason": "gold_index_not_found_on_disk",
            "expected_path": _default_gold_rag_build_dir(),
            "kaggle_dataset": "salmashopna/rag-index-gold",
            "kaggle_dataset_path": KAGGLE_GOLD_RAG_DATASET_DIR,
            "searched_paths": _gold_rag_dir_candidates()[:8],
            "paper_note": (
                "Attach Kaggle dataset salmashopna/rag-index-gold, or run --build_gold_index, "
                "then --auto_gold_pubmedqa for PubMedQA recall@k."
            ),
        }
    os.environ["GP_RAG_GOLD_INDEX_DIR"] = cand
    print(
        f"\n{'=' * 60}\n"
        f"AUTO GOLD PUBMEDQA: corpus_mode=dual_primary_plus_gold_pubmedqa\n"
        f"  GP_RAG_GOLD_INDEX_DIR={cand!r}\n"
        f"  (primary external index unchanged; MCQ vs PubMedQA retrieval are separate experiments.)\n"
        f"{'=' * 60}\n",
        flush=True,
    )
    return {
        "auto_gold_enabled": True,
        "gold_dir": cand,
        "corpus_mode": "dual_primary_plus_gold_pubmedqa",
        "opt_out": "GP_DISABLE_AUTO_GOLD_PUBMEDQA=1",
    }


def _chunk_ids_from_rag(
    rag_hits: List[Dict[str, Any]],
    ranked_sources: List[str],
    diag: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Ordered chunk/source ids for CSV diagnostics and recall@k."""
    d = diag or {}
    for key in ("retrieved_chunk_ids", "final_chunk_ids"):
        raw = d.get(key)
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw if str(x)]
    out: List[str] = []
    seen: Set[str] = set()
    for h in rag_hits:
        if not isinstance(h, dict):
            continue
        s = str(h.get("source") or h.get("chunk_id") or "").strip()
        if not s and h.get("idx") is not None:
            s = f"idx_{h['idx']}"
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for s in ranked_sources:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


_MCQ_FINAL_ANSWER_SUFFIX = "\n\nFinal answer letter:"


def _mcq_gen_max_new_tokens(model_key: str, effective_rag: bool) -> int:
    """Longer generation when RAG context is in-prompt (LLM especially)."""
    if effective_rag and model_key == "llm":
        raw = os.environ.get("LLM_RAG_MCQ_MAX_NEW_TOKENS", "64").strip()
    elif effective_rag:
        raw = os.environ.get("RAG_MCQ_MAX_NEW_TOKENS", "48").strip()
    else:
        raw = os.environ.get("MCQ_MAX_NEW_TOKENS", "32").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 64 if effective_rag and model_key == "llm" else (48 if effective_rag else 32)
    return max(16, min(128, n))


def _build_mcq_prompt(
    question: str,
    choices: List[str],
    context: str = "",
    use_rag: bool = False,
) -> str:
    """Question + options last; optional reference block first when RAG is used."""
    options_block = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    tail = (
        "Evaluate all four options (A, B, C, and D) equally before deciding.\n"
        "Reply with only the single letter of the best option."
        f"{_MCQ_FINAL_ANSWER_SUFFIX}"
    )
    if use_rag and context and len(context.strip()) > 50:
        return (
            "Use the following medical reference to help answer the question.\n\n"
            f"Reference:\n{context.strip()}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{tail}"
        )
    return f"Question: {question}\n\nOptions:\n{options_block}\n\n{tail}"


def _pubmedqa_prompt_with_optional_rag(
    question: str,
    abstract: str,
    rag_block: str,
    model_key: str = "",
) -> str:
    base = _pubmedqa_user_prompt(question, abstract, model_key=model_key)
    if rag_block and len(rag_block.strip()) > 50:
        return (
            "Additional retrieved references (use if relevant to the abstract):\n"
            f"{rag_block.strip()}\n\n"
            f"{base}"
        )
    return base


def _mcq_token_f1(
    pred_letter: str,
    gold_letter: str,
    choices: List[str],
) -> Tuple[float, float]:
    """(token_f1 on full option text, token_f1_label exact letter match)."""
    pred_text = _mcq_choice_text(choices, pred_letter)
    gold_text = _mcq_choice_text(choices, gold_letter)
    label_match = (
        1.0
        if pred_letter and gold_letter and pred_letter.strip().upper() == gold_letter.strip().upper()
        else 0.0
    )
    if not pred_text or not gold_text:
        return 0.0, label_match
    return _token_f1(pred_text, gold_text), label_match


def _pubmedqa_token_f1(
    raw_response: str,
    parsed_label: str,
    ref_label: str,
    ref_long_answer: str,
) -> Tuple[float, float]:
    """(token_f1 vs long_answer, token_f1_label vs yes/no/maybe)."""
    ref_text = ref_long_answer if ref_long_answer else ref_label
    f1_main = _token_f1(raw_response or "", ref_text) if ref_text else 0.0
    f1_label = _token_f1(parsed_label or "", ref_label) if ref_label else 0.0
    return f1_main, f1_label


def _validate_token_f1_distribution(rows: List[Dict[str, Any]], benchmark: str) -> None:
    """Warn if token_f1 collapsed to binary 0/1 only (exact-match regression)."""
    f1_vals = [
        float(r["token_f1"])
        for r in rows
        if r.get("benchmark") == benchmark
        and r.get("token_f1") is not None
        and str(r.get("token_f1")).strip() not in ("", "nan", "None")
    ]
    if not f1_vals:
        return
    unique_vals = {round(v, 3) for v in f1_vals}
    binary_only = unique_vals.issubset({0.0, 1.0})
    near_zero_pct = sum(1 for v in f1_vals if v < 0.01) / len(f1_vals)
    if binary_only or near_zero_pct > 0.85:
        print(
            f"\nWARNING [{benchmark}]: token_f1 looks binary "
            f"(unique={sorted(unique_vals)[:10]}, {near_zero_pct:.0%} near-zero). "
            "Expected fractional overlap when option/long_answer text differs.",
            flush=True,
        )
    else:
        print(
            f"token_f1 OK [{benchmark}]: mean={sum(f1_vals) / len(f1_vals):.3f}, "
            f"n_unique={len(unique_vals)}, n={len(f1_vals)}",
            flush=True,
        )


def _validate_retrieval_labels_before_run(
    benchmarks: List[str],
    rag_index_dir: str,
    gold_index_dir: str = "",
) -> Dict[str, Any]:
    """Pre-flight: PubMedQA recall@k requires pubmedqa_* chunks in the retrieval corpus."""
    if "pubmedqa" not in benchmarks:
        return {"retrieval_validation": "skipped", "reason": "no_pubmedqa_in_run"}
    gold_dir = (gold_index_dir or _pubmedqa_rag_index_dir()).strip()
    if gold_dir and _gold_rag_index_ready(gold_dir):
        gcp = os.path.join(os.path.abspath(gold_dir), "chunks.jsonl")
        n_total, _is_seed, pubmed_hits = _rag_chunks_metadata(gcp)
        primary = ""
        if rag_index_dir:
            primary = os.path.join(os.path.abspath(rag_index_dir), "chunks.jsonl")
        print(
            f"\nRetrieval eval: PubMedQA uses gold index {gold_dir!r} "
            f"({pubmed_hits} pubmedqa_* chunks). Primary MCQ index unchanged.\n",
            flush=True,
        )
        split = _ensure_pubmedqa_split_loaded()
        eval_ids = _pubmedqa_eval_pubids()
        missing_eval = _gold_index_missing_eval_pubids(gold_dir, eval_ids)
        rq = (
            "question_plus_abstract"
            if _pubmedqa_retrieval_query_includes_abstract()
            else "question_only"
        )
        if missing_eval:
            print(
                f"\nWARNING: gold index missing {len(missing_eval)} eval-holdout PMIDs "
                f"(recall@k will be 0 for those). Rebuild:\n"
                f"  python eval_benchmarks.py --build_gold_index {gold_dir!r}\n",
                flush=True,
            )
        else:
            print(
                f"\nRetrieval eval OK: gold index covers all {len(eval_ids)} eval-holdout PMIDs "
                f"(recall@k meaningful with question-only queries).\n",
                flush=True,
            )
        return {
            "retrieval_validation": "ok" if not missing_eval else "invalid_missing_eval_pubids",
            "corpus_mode": "dual_primary_plus_gold_pubmedqa",
            "gold_chunks_path": gcp,
            "primary_chunks_path": primary or None,
            "n_total_chunks": n_total,
            "n_pubmedqa_gold_chunks": pubmed_hits,
            "n_eval_holdout_pubids": len(eval_ids),
            "n_eval_pubids_missing_from_gold_index": len(missing_eval),
            "recall_at_k_meaningful": not missing_eval,
            "pubmedqa_eval_split": split.get("eval_split_descriptor"),
            "pubmedqa_retrieval_query_mode": rq,
            "task_rag_uses_primary_external": True,
        }
    chunks_path = ""
    if rag_index_dir:
        chunks_path = os.path.join(os.path.abspath(rag_index_dir), "chunks.jsonl")
    if not chunks_path or not os.path.isfile(chunks_path):
        cp = _active_rag_chunks_jsonl()
        chunks_path = cp or chunks_path
    if not chunks_path or not os.path.isfile(chunks_path):
        return {
            "retrieval_validation": "failed",
            "reason": "chunks_jsonl_not_found",
            "recommendation": (
                "python eval_benchmarks.py --build_gold_index /kaggle/working/rag_index_gold "
                "then --auto_gold_pubmedqa or --rag_gold_index_dir <dir>"
            ),
        }
    n_total, _is_seed, pubmed_hits = _rag_chunks_metadata(chunks_path)
    evaluable = pubmed_hits > 0
    result: Dict[str, Any] = {
        "retrieval_validation": "ok" if evaluable else "invalid",
        "corpus_mode": "single_primary",
        "chunks_path": chunks_path,
        "n_total_chunks": n_total,
        "n_pubmedqa_gold_chunks": pubmed_hits,
        "recall_at_k_meaningful": evaluable,
    }
    if not evaluable:
        print(
            f"\n{'=' * 60}\n"
            "RETRIEVAL EVALUATION INVALID\n"
            f"Corpus {chunks_path!r} has {n_total} chunks but ZERO pubmedqa_* gold chunks.\n"
            "Recall@K will be 0 for all PubMedQA RAG rows — do not report as retrieval performance.\n"
            "Fix:\n"
            "  python eval_benchmarks.py --build_gold_index /kaggle/working/rag_index_gold\n"
            "  python eval_benchmarks.py --benchmark all --auto_gold_pubmedqa --seed 42\n"
            f"{'=' * 60}\n",
            flush=True,
        )
        result["recommendation"] = (
            "run --build_gold_index OUT_DIR (FAISS included), then --auto_gold_pubmedqa"
        )
        result["audit_m13_note"] = (
            "recall@k=0 on external corpus is expected (no pubmedqa_* gold chunks); "
            "do not cite as retriever failure. Use gold index for PubMedQA retrieval metrics."
        )
    return result


def _check_rag_accuracy_delta(
    benchmark_results: Dict[str, Any],
    benchmark_name: str,
) -> Dict[str, Any]:
    """Warn when RAG consistently hurts accuracy vs NoRAG."""
    diagnostics: Dict[str, Any] = {}
    for model_prefix in ("SLM", "LLM"):
        no_rag = benchmark_results.get(f"{model_prefix}_NoRAG", {})
        rag = benchmark_results.get(f"{model_prefix}_RAG", {})
        no_rag_acc = no_rag.get("mean") or no_rag.get("pubmedqa_label_accuracy_mean")
        rag_acc = rag.get("mean") or rag.get("pubmedqa_label_accuracy_mean")
        if no_rag_acc is None or rag_acc is None:
            continue
        delta = float(rag_acc) - float(no_rag_acc)
        diagnostics[f"{model_prefix}_rag_delta"] = round(delta, 4)
        if delta < -0.05:
            diagnostics[f"{model_prefix}_rag_harmful"] = True
            print(
                f"\nWARNING [{benchmark_name} / {model_prefix}]: RAG hurts accuracy by "
                f"{abs(delta):.1%} (NoRAG={no_rag_acc:.3f}, RAG={rag_acc:.3f}).\n"
                "  Check rag_context_rejected in predictions; use gold index + RAG_MIN_SCORE.\n",
                flush=True,
            )
        elif delta > 0.02:
            print(
                f"OK [{benchmark_name} / {model_prefix}]: RAG improves accuracy by "
                f"{delta:.1%} (NoRAG={no_rag_acc:.3f}, RAG={rag_acc:.3f}).",
                flush=True,
            )
    return diagnostics


# Seeded bootstrap RNG (audit M2); set in ``run_all_benchmarks`` from ``subset_seed``.
_CI_RNG: Optional[random.Random] = None


def _set_bootstrap_rng(subset_seed: Optional[int]) -> None:
    global _CI_RNG
    if subset_seed is not None:
        _CI_RNG = random.Random(int(subset_seed) + 17)
    else:
        _CI_RNG = random.Random()


def _bootstrap_rng() -> random.Random:
    return _CI_RNG if _CI_RNG is not None else random


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant pairs (b=NoRAG wrong/RAG right, c=opposite)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    # Two-sided: 2 * min(P(X<=k), P(X>=k)) capped at 1
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5**n)
    p_one = prob
    p_two = min(1.0, 2.0 * p_one)
    return p_two


def _paired_bootstrap_mean_ci(
    values: List[float], alpha: float = 0.05, iters: int = 2000
) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = _bootstrap_rng()
    n = len(values)
    mean = sum(values) / n
    samples: List[float] = []
    for _ in range(iters):
        samples.append(sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n)
    samples.sort()
    lo = int((alpha / 2) * iters)
    hi = int((1 - alpha / 2) * iters)
    return mean, samples[lo], samples[min(hi, iters - 1)]


def _row_task_correct(row: Dict[str, Any]) -> Optional[bool]:
    mcq = row.get("mcq_correct")
    if mcq is True or mcq is False:
        return bool(mcq)
    if mcq in (1, 1.0):
        return True
    if mcq in (0, 0.0):
        return False
    lab = row.get("label_correct")
    if lab is True or lab is False:
        return bool(lab)
    if lab in (1, 1.0):
        return True
    if lab in (0, 0.0):
        return False
    return None


def _paired_rag_significance(
    rows: List[Dict[str, Any]], benchmark_name: str
) -> Dict[str, Any]:
    """Paired McNemar + bootstrap delta for RAG vs NoRAG on the same items (audit M1)."""
    by_item: Dict[Tuple[str, str], Dict[str, bool]] = {}
    for r in rows:
        if str(r.get("benchmark")) != benchmark_name:
            continue
        qid = str(r.get("question_id") or "")
        mk = str(r.get("model_key") or "")
        cfg = str(r.get("model_name") or "")
        ok = _row_task_correct(r)
        if ok is None or not qid or not mk:
            continue
        key = (qid, mk)
        slot = by_item.setdefault(key, {})
        if cfg.endswith("_NoRAG"):
            slot["no"] = ok
        elif cfg.endswith("_RAG"):
            slot["rag"] = ok

    out: Dict[str, Any] = {"benchmark": benchmark_name, "n_paired": 0}
    for model_prefix, mk in (("SLM", "slm"), ("LLM", "llm")):
        paired: List[Tuple[bool, bool]] = []
        for (_qid, key_mk), slot in by_item.items():
            if key_mk != mk:
                continue
            if "no" in slot and "rag" in slot:
                paired.append((slot["no"], slot["rag"]))
        if not paired:
            continue
        b = sum(1 for no, rag in paired if (not no) and rag)
        c = sum(1 for no, rag in paired if no and (not rag))
        deltas = [float(rag) - float(no) for no, rag in paired]
        dm, dlo, dhi = _paired_bootstrap_mean_ci(deltas)
        prefix = model_prefix.lower()
        out[f"{prefix}_n_paired"] = len(paired)
        out[f"{prefix}_mcnemar_b_rag_wins"] = b
        out[f"{prefix}_mcnemar_c_rag_loses"] = c
        out[f"{prefix}_mcnemar_p_value"] = round(_mcnemar_exact_p(b, c), 6)
        out[f"{prefix}_accuracy_delta_mean"] = round(dm, 4)
        out[f"{prefix}_accuracy_delta_ci_lower"] = round(dlo, 4)
        out[f"{prefix}_accuracy_delta_ci_upper"] = round(dhi, 4)
        out["n_paired"] = max(int(out.get("n_paired") or 0), len(paired))
    return out


def _paired_rag_significance_all(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    benches = sorted({str(r.get("benchmark")) for r in rows if r.get("benchmark")})
    return {b: _paired_rag_significance(rows, b) for b in benches}


def _accuracy_cluster_bootstrap(
    preds: List[Any], golds: List[Any], cluster_ids: List[str]
) -> Tuple[float, float, float]:
    """Cluster bootstrap by subject (audit M9 — MMLU-Med macro subjects)."""
    if not preds or len(preds) != len(golds) or len(preds) != len(cluster_ids):
        return float("nan"), float("nan"), float("nan")
    clusters: Dict[str, List[float]] = {}
    for p, g, cid in zip(preds, golds, cluster_ids):
        clusters.setdefault(str(cid), []).append(1.0 if p == g else 0.0)
    cluster_means = [sum(v) / len(v) for v in clusters.values() if v]
    if not cluster_means:
        return float("nan"), float("nan"), float("nan")
    rng = _bootstrap_rng()
    iters = 2000
    alpha = 0.05
    n_c = len(cluster_means)
    samples: List[float] = []
    for _ in range(iters):
        draw = [cluster_means[rng.randint(0, n_c - 1)] for _ in range(n_c)]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    mean = sum(cluster_means) / n_c
    lo = int((alpha / 2) * iters)
    hi = int((1 - alpha / 2) * iters)
    return mean, samples[lo], samples[min(hi, iters - 1)]


def _dataset_provenance_snapshot(benchmarks: List[str]) -> Dict[str, Any]:
    """Pinned dataset IDs and eval protocol notes (audit M4)."""
    prov: Dict[str, Any] = {
        "medqa": {
            "hub_id": "nnilayy/medqa-usmle",
            "split": "validation",
            "canonical_note": (
                "Not bigbio/med_qa; field mapping uses sent1/sent2/ending*. "
                "Pin HF revision in paper if comparing to literature."
            ),
        },
        "mmlu_med": {
            "hub_id": "cais/mmlu",
            "subjects": list(MMLU_MED_SUBJECTS),
            "split": _mmlu_med_split(),
            "contamination_note": "dev split default; test may overlap pretraining.",
        },
        "pubmedqa": {
            "hub_id": "pubmed_qa",
            "config": "pqa_labeled",
            "hf_split_loaded": PUBMEDQA_LABELED_SPLIT,
            "eval_protocol": _pubmedqa_eval_split_descriptor(),
            "not_official_pubmedqa_test": True,
        },
    }
    return {k: prov[k] for k in benchmarks if k in prov}


def _helper_bundle_hashes(work_dir: str) -> Dict[str, Any]:
    """SHA-256 of helper modules on disk (audit M7)."""
    import hashlib

    names = list(_HELPER_MODULES) + ["rag_retrieval.py"]
    out: Dict[str, str] = {}
    for name in names:
        path = os.path.join(work_dir, name)
        if not os.path.isfile(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        out[name] = h.hexdigest()
    out["gp_bundle_version"] = GP_BUNDLE_VERSION
    return out


def _rag_context_rejection_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of RAG-flagged rows where context was rejected (audit M6/m6)."""
    summary: Dict[str, Any] = {}
    for cfg in [c[0] for c in CONFIGS if c[2]]:
        sub = [r for r in rows if str(r.get("model_name")) == cfg]
        if not sub:
            continue
        n = len(sub)
        rej = sum(1 for r in sub if r.get("rag_context_rejected") in (True, 1, 1.0))
        summary[cfg] = {
            "n_rag_rows": n,
            "n_context_rejected": rej,
            "pct_context_rejected": round(100.0 * rej / n, 2) if n else 0.0,
        }
    return summary


def _evidence_token_overlap(context: str, answer: str) -> float:
    """Minimal faithfulness proxy: token Jaccard between evidence and answer (audit M10)."""
    def _tok(s: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}

    a, b = _tok(context), _tok(answer)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pubmedqa_parse_status(raw: str, parsed: str) -> str:
    """audit m2: ok | unparseable | empty_response."""
    if not str(raw or "").strip():
        return "empty_response"
    p = str(parsed or "").strip().lower()
    if p in ("yes", "no", "maybe"):
        return "ok"
    return "unparseable"


def _pubmedqa_parse_failure_rate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sub = [r for r in rows if str(r.get("benchmark")) == "pubmedqa"]
    if not sub:
        return {}
    n = len(sub)
    counts: Dict[str, int] = {"ok": 0, "unparseable": 0, "empty_response": 0}
    for r in sub:
        st = str(r.get("pubmedqa_parse_status") or "")
        if not st:
            st = _pubmedqa_parse_status(
                str(r.get("raw_response") or ""),
                str(r.get("parsed_prediction") or ""),
            )
        counts[st] = counts.get(st, 0) + 1
    unparseable = counts.get("unparseable", 0)
    return {
        "n_rows": n,
        "n_ok": counts.get("ok", 0),
        "n_unparseable": unparseable,
        "n_empty_response": counts.get("empty_response", 0),
        "pct_unparseable": round(100.0 * unparseable / n, 2) if n else 0.0,
        "by_status": counts,
    }


def _energy_measurement_disclosure() -> Dict[str, Any]:
    """audit m4: runner token heuristic is not wall-power; cite measurement_config for paper."""
    out: Dict[str, Any] = {
        "runner_energy_field": "energy_kwh_per_token_heuristic",
        "not_in_benchmark_json_by_default": True,
        "paper_recommendation": "Use NVML / measurement_config per-query kWh for energy claims, not runner heuristics.",
    }
    try:
        import measurement_config as mc

        out["measurement_config_slugs"] = {
            "SLM_NoRAG": getattr(mc, "ENERGY_KWH_PER_QUERY_SLM_NORAG", None),
            "SLM_RAG": getattr(mc, "ENERGY_KWH_PER_QUERY_SLM_RAG", None),
            "LLM_NoRAG": getattr(mc, "ENERGY_KWH_PER_QUERY_LLM_NORAG", None),
            "LLM_RAG": getattr(mc, "ENERGY_KWH_PER_QUERY_LLM_RAG", None),
        }
    except Exception as ex:
        out["measurement_config_error"] = str(ex)
    return out


def _rag_context_rejection_by_benchmark(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """audit m6: rejection rate split by benchmark."""
    out: Dict[str, Any] = {}
    for bench in sorted({str(r.get("benchmark")) for r in rows if r.get("benchmark")}):
        sub = [r for r in rows if str(r.get("benchmark")) == bench and r.get("rag_flag")]
        if not sub:
            continue
        n = len(sub)
        rej = sum(1 for r in sub if r.get("rag_context_rejected") in (True, 1, 1.0))
        out[bench] = {
            "n_rag_rows": n,
            "n_context_rejected": rej,
            "pct_context_rejected": round(100.0 * rej / n, 2) if n else 0.0,
        }
    return out


def _enforce_strict_metrics(agg: Dict[str, Any]) -> None:
    """audit m7: fail closed when primary metrics are NaN."""
    bad: List[str] = []
    for bench, entry in (agg.get("benchmarks") or {}).items():
        if not isinstance(entry, dict):
            continue
        for cfg, res in (entry.get("results") or {}).items():
            if not isinstance(res, dict):
                continue
            for key in (
                "mean",
                "pubmedqa_label_accuracy_mean",
                "f1_mean",
                "recall_at_1_mean",
            ):
                val = res.get(key)
                if val is None:
                    continue
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                if fv != fv:
                    bad.append(f"{bench}/{cfg}/{key}=NaN")
    if bad:
        raise RuntimeError(
            "GP_BENCH strict metrics: missing or failed metric computations:\n  "
            + "\n  ".join(bad[:20])
            + ("\n  ..." if len(bad) > 20 else "")
            + "\nInstall rouge-score, sacrebleu, nltk, bert-score or disable generative metrics."
        )


def _multiple_comparisons_disclosure(benchmarks: List[str]) -> Dict[str, Any]:
    """Pre-registered contrasts + FDR note (audit M12)."""
    n_cells = len(CONFIGS) * len(benchmarks)
    return {
        "n_config_benchmark_cells": n_cells,
        "bonferroni_alpha_0_05_if_all_primary": round(0.05 / max(n_cells, 1), 6),
        "recommended_correction": "Apply FDR (Benjamini-Hochberg) across exploratory contrasts only.",
        "primary_contrasts": [
            {"benchmark": b, "contrast": "LLM_RAG vs LLM_NoRAG"}
            for b in benchmarks
        ],
        "secondary_contrasts": [
            {"benchmark": b, "contrast": "SLM_RAG vs SLM_NoRAG"}
            for b in benchmarks
        ],
    }


def _rag_recall_ks() -> Tuple[int, ...]:
    raw = os.environ.get("RAG_RECALL_K", "1,3,5,10").strip()
    ks: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ks.append(max(1, int(part)))
    return tuple(sorted(set(ks))) if ks else (1, 3, 5, 10)


def _rag_relevant_source_ids(item: Dict[str, Any], benchmark: str) -> Set[str]:
    """
    Gold relevant chunk id(s) for retrieval metrics.

    Matches ``pubmedqa_<pubid>`` and ``pubmedqa_<pubid>_<chunk>`` via prefix rules in
    ``eval_quality_metrics.doc_matches_relevant``.
    """
    if benchmark != "pubmedqa":
        return set()
    qid = str(item.get("id") or "").strip()
    if not qid:
        return set()
    rel = {f"pubmedqa_{qid}", f"pubmedqa_{qid}_long"}
    return rel


def _model_ids_snapshot(hf_router_meta: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Resolved Hugging Face / router model ids for reproducibility."""
    if hf_router_meta:
        slm = str(hf_router_meta.get("model_slm") or "")
        llm = str(hf_router_meta.get("model_llm") or "")
    else:
        try:
            _setup_runner_path()
            import real_model_runner as rmr

            slm = str(rmr.MODEL_SLM)
            llm = str(rmr.MODEL_LLM)
        except Exception:
            slm = os.environ.get("GP_MODEL_SLM", "google/gemma-2-2b-it").strip()
            llm = os.environ.get("GP_MODEL_LLM", "meta-llama/Llama-2-7b-chat-hf").strip()
    return {"slm_model_id": slm, "llm_model_id": llm}


def _row_model_meta(model_key: str, model_ids: Dict[str, str]) -> Dict[str, str]:
    inf = model_ids["slm_model_id"] if model_key == "slm" else model_ids["llm_model_id"]
    return {
        "slm_model_id": model_ids["slm_model_id"],
        "llm_model_id": model_ids["llm_model_id"],
        "inference_model_id": inf,
    }


def _ranked_sources_for_metrics(
    ranked_sources: List[str],
    rag_hits: List[Dict[str, Any]],
    rag_diagnostic: Optional[Dict[str, Any]] = None,
) -> List[str]:
    out = _enrich_ranked_sources_from_diag(
        [str(s) for s in ranked_sources if str(s)],
        rag_diagnostic,
    )
    if out:
        return out
    for h in rag_hits:
        if isinstance(h, dict):
            s = str(h.get("source") or "").strip()
            if s:
                out.append(s)
    return out


def _retrieval_metrics_fields(
    benchmark: str,
    item: Dict[str, Any],
    use_rag: bool,
    ranked_sources: List[str],
    rag_hits: Optional[List[Dict[str, Any]]] = None,
    rag_diagnostic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not use_rag:
        return {}
    rel = _rag_relevant_source_ids(item, benchmark)
    if not rel:
        return {"retrieval_evaluable": False}
    ranked_sources = _ranked_sources_for_metrics(
        ranked_sources, rag_hits or [], rag_diagnostic
    )
    from eval_quality_metrics import compute_retrieval_metrics

    m = compute_retrieval_metrics(ranked_sources, rel, ks=_rag_recall_ks())
    m["retrieval_evaluable"] = True
    m["rag_relevant_sources"] = sorted(rel)
    if _pubmedqa_retrieval_index_dir() and benchmark == "pubmedqa":
        m["rag_gold_index_dir"] = _pubmedqa_retrieval_index_dir()
    return m


_RETRIEVAL_NA_LOG_MSG = (
    "recall@k = 0 is expected with the external non-leaking corpus — no pubmedqa_* gold chunks present. "
    "Retrieval metrics are N/A for this index. Use a gold-aligned index to evaluate recall@k meaningfully."
)

_RETRIEVAL_NA_SUMMARY_MSG = (
    "Retrieval eval N/A: external non-leaking corpus has no pubmedqa_* gold chunks. "
    "This is expected and correct — rebuild with a gold-aligned index only if you need recall@k numbers."
)


def _active_rag_chunks_jsonl() -> str:
    cp = os.environ.get("RAG_CHUNKS_JSONL", "").strip()
    if cp and os.path.isfile(cp):
        return os.path.abspath(cp)
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    if d:
        p = os.path.join(os.path.abspath(d), "chunks.jsonl")
        if os.path.isfile(p):
            return p
    return ""


def _rag_corpus_pubmed_hits() -> int:
    """Count of ``pubmedqa_*`` lines in the active RAG chunks file (-1 if unknown)."""
    cp = _active_rag_chunks_jsonl()
    if not cp:
        return -1
    _n, _is_seed, pubmed_hits = _rag_chunks_metadata(cp)
    return int(pubmed_hits)


def _row_recalls_all_zero(eval_rows: List[Dict[str, Any]], ks: Tuple[int, ...]) -> bool:
    if not eval_rows:
        return True
    for r in eval_rows:
        for k in ks:
            key = f"recall_at_{k}"
            if key in r and r[key] == r[key] and float(r[key]) > 1e-9:
                return False
    return True


def _recall_means_all_zero(means: Dict[str, Any], ks: Tuple[int, ...]) -> bool:
    for k in ks:
        v = means.get(f"recall_at_{k}_mean")
        if v is not None and v == v and float(v) > 1e-9:
            return False
    return True


def _recall_means_any_positive(means: Dict[str, Any], ks: Tuple[int, ...]) -> bool:
    for k in ks:
        v = means.get(f"recall_at_{k}_mean")
        if v is not None and v == v and float(v) > 1e-9:
            return True
    return False


def _external_corpus_retrieval_metrics_na(pubmed_hits: Optional[int] = None) -> bool:
    if _pubmedqa_gold_rag_active():
        return False
    if pubmed_hits is None:
        pubmed_hits = _rag_corpus_pubmed_hits()
    return pubmed_hits == 0


def _config_retrieval_summary(cfg_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Macro-average Recall@K and MRR over evaluable rows for one model config."""
    eval_rows = [r for r in cfg_rows if r.get("retrieval_evaluable")]
    if not eval_rows:
        return {}
    ks = _rag_recall_ks()
    if _external_corpus_retrieval_metrics_na() and _row_recalls_all_zero(eval_rows, ks):
        return {
            "retrieval_metrics_na": True,
            "retrieval_n_evaluable": len(eval_rows),
            "retrieval_na_reason": "external_non_leaking_corpus_no_pubmedqa_gold",
        }
    out: Dict[str, Any] = {"retrieval_n_evaluable": len(eval_rows)}
    for k in ks:
        key = f"recall_at_{k}"
        vals = [float(r[key]) for r in eval_rows if key in r and r[key] == r[key]]
        if vals:
            m, lo, hi = _bootstrap_ci(vals)
            out[f"{key}_mean"] = m
            out[f"{key}_ci_lower"] = lo
            out[f"{key}_ci_upper"] = hi
    mrr_vals = [float(r["mrr"]) for r in eval_rows if "mrr" in r and r["mrr"] == r["mrr"]]
    if mrr_vals:
        m, lo, hi = _bootstrap_ci(mrr_vals)
        out["mrr_mean"] = m
        out["mrr_ci_lower"] = lo
        out["mrr_ci_upper"] = hi
    return out


_DEFAULT_HF_ROUTER_LLM = "meta-llama/Llama-2-7b-chat-hf"


def _resolve_hf_router_models(cli_slm: str, cli_llm: str) -> Tuple[str, str]:
    llm = (cli_llm or os.environ.get("HF_ROUTER_MODEL_LLM", "") or "").strip() or _DEFAULT_HF_ROUTER_LLM
    slm = (cli_slm or os.environ.get("HF_ROUTER_MODEL_SLM", "") or "").strip() or llm
    return slm, llm


def _hf_router_max_tokens(pubmed_one_word: bool = False) -> int:
    if pubmed_one_word:
        return _pubmedqa_max_new_tokens()
    raw = os.environ.get("GEN_MAX_NEW_TOKENS", "").strip()
    v = int(raw) if raw.isdigit() else 32
    return max(16, min(1024, v))


def _nonempty_env(*keys: str) -> bool:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip():
            return True
    return False


def _try_load_hf_token() -> bool:
    """Load HF token from env, hub cache, Kaggle Secrets, or /kaggle/secret/* (no raise)."""
    if _nonempty_env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        t = get_token()
        if t and str(t).strip():
            os.environ.setdefault("HF_TOKEN", str(t).strip())
            print("Loaded HF token from huggingface_hub cache (get_token).", flush=True)
            return True
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
                    return True
            except Exception:
                continue
    except Exception:
        pass
    for p in ("/kaggle/secret/hf_token", "/kaggle/secret/HF_TOKEN"):
        if os.path.isfile(p):
            os.environ["HF_TOKEN"] = open(p, encoding="utf-8").read().strip()
            print(f"Loaded HF_TOKEN from {p}.", flush=True)
            return True
    return False


def _ensure_hf_token() -> None:
    if _try_load_hf_token():
            return

    if os.environ.get("GP_BENCH_ALLOW_NO_HF_TOKEN", "").strip().lower() in ("1", "true", "yes", "on"):
        print(
            "GP_BENCH_ALLOW_NO_HF_TOKEN: continuing without HF_TOKEN (public Hub only; "
            "set HF_TOKEN for gated models/datasets).",
            flush=True,
        )
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
        hint += (
            "\nLocally: export HF_TOKEN=... or huggingface-cli login.\n"
            "Optional: GP_BENCH_ALLOW_NO_HF_TOKEN=1 to skip this check for public Hub models/datasets only.\n"
        )
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
    """
    If max_items <= 0, use full list.
    If seed is None, take the first max_items in dataset order (--first_n).
    Else shuffle with seed and take max_items (different draws per seed; default seed is random each run).
    """
    if max_items <= 0 or max_items >= len(items):
        return items
    if seed is None:
        return items[:max_items]
    rng = random.Random(int(seed))
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:max_items]]


def _resolve_subset_seed(max_items: int, seed: Optional[int], first_n: bool) -> Optional[int]:
    """Pick a random seed per run when capping items, unless --seed or --first_n is set."""
    if max_items <= 0 or first_n:
        return seed
    if seed is not None:
        return int(seed)
    return secrets.randbelow(2**31)


_LOAD_DATASET_FN: Any = None

# Kaggle image ships pandas 2.x; do not ``pip install pandas>=2.0`` without ``<3.0`` (breaks cudf/gradio).
_KAGGLE_PANDAS_SPEC = "pandas>=2.0,<3.0"
# Install numpy + scipy in the SAME pip line as pandas (force-reinstall breaks scipy↔numpy otherwise).
_KAGGLE_NUMPY_SCIPY_SPEC_PY312 = (
    "numpy>=1.26.4,<2.1",
    "scipy>=1.11.4,<1.15",
)
_KAGGLE_HF_STACK_SPECS = (
    "transformers>=4.43.0,<5",
    "huggingface-hub>=0.23.0",
    "accelerate>=0.26.0",
    "sentence-transformers>=2.2.2",
    "safetensors>=0.4.0",
    "faiss-cpu",
)

_PREDICTION_CSV_COLUMNS = [
    "benchmark",
    "question_id",
    "question",
    "reference_answer",
    "reference_text",
    "prediction_text",
    "context",
    "context_chars",
    "model_name",
    "model_key",
    "slm_model_id",
    "llm_model_id",
    "inference_model_id",
    "rag_flag",
    "rag_context_used",
    "rag_context_rejected",
    "model_answer",
    "raw_response",
    "retrieved_context",
    "rag_source",
    "rag_hits",
    "rag_ranked_sources",
    "parsed_prediction",
    "label_correct",
    "mcq_correct",
    "choices_json",
    "token_f1",
    "token_f1_label",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "retrieval_method",
    "retrieved_chunk_ids",
    "similarity_scores",
    "reranker_scores",
    "gold_chunk_rank",
    "gold_chunk_found",
    "retrieval_latency_ms",
    "latency_seconds",
    "response_tokens",
    "energy_joules_nvml",
    "gpu_energy_kwh_nvml",
    "gpu_power_watts_mean_nvml",
    "gpu_name",
    "quantization_backend",
]

_KAGGLE_NUMPY_SCIPY_SPEC_PY312_NP2 = (
    "numpy>=2.0,<2.2",
    "scipy>=1.14.1",
)


def _quiet_hf_datasets() -> None:
    """
    Suppress Hugging Face / tqdm progress widgets (ChunkLoadError: HBoxModel is harmless).

    Call before ``load_dataset``; set in notebook Cell 1 if an earlier cell imported ``datasets``.
    """
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from datasets import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass
    try:
        from datasets import logging as ds_logging

        ds_logging.set_verbosity_error()
    except Exception:
        pass


def _require_pandas2_on_kaggle(*, allow_pandas3: bool = False) -> None:
    """
    Abort on Kaggle if pandas 3.x is active (invalidates ``datasets`` / benchmark results).

    Fix: use the numpy+scipy+pandas pip line from the module docstring, then Kernel → Restart Session.
    """
    if not os.path.isdir("/kaggle"):
        return
    if allow_pandas3 or os.environ.get("GP_BENCH_ALLOW_PANDAS3", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        import pandas as pd

        ver = str(getattr(pd, "__version__", "0"))
        major = int(ver.split(".")[0])
    except Exception as ex:
        raise RuntimeError(
            "Could not import pandas on Kaggle. Run Cell 1: "
            f"pip install -q '{_KAGGLE_PANDAS_SPEC}' --force-reinstall, then restart kernel."
        ) from ex
    if major >= 3:
        raise RuntimeError(
            f"pandas {ver} on Kaggle breaks Hugging Face datasets and invalidates this run. "
            "Run as your FIRST cell, then Kernel → Restart Session:\n"
            f"  !pip install -q '{_KAGGLE_PANDAS_SPEC}' --force-reinstall\n"
            "Verify: import pandas as pd; print(pd.__version__)  # must be 2.x\n"
            "Or pass --allow_pandas3 to override (not recommended)."
        )
    print(f"Kaggle: pandas {ver} OK (2.x required).", flush=True)


def _kaggle_numpy_scipy_pip_line() -> str:
    specs = list(_KAGGLE_NUMPY_SCIPY_SPEC_PY312) + [_KAGGLE_PANDAS_SPEC, "datasets>=2.14", "pyarrow"]
    return "pip install -q " + " ".join(f"'{s}'" for s in specs) + " --force-reinstall"


def _kaggle_hf_stack_pip_line() -> str:
    specs = list(_KAGGLE_HF_STACK_SPECS) + [
        "bitsandbytes>=0.46.1",
        "rouge-score",
        "sacrebleu",
        "nltk",
        "bert-score",
        "openai",
    ]
    return "pip install -q " + " ".join(f"'{s}'" for s in specs) + " --upgrade"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _running_in_ipython_kernel() -> bool:
    return "ipykernel" in sys.modules


def _kaggle_strict_checks() -> bool:
    return _env_truthy("GP_BENCH_STRICT")


def _kaggle_reexec_argv_tail() -> List[str]:
    """CLI args for subprocess re-exec (strip notebook -f kernel.json / ipykernel paths)."""
    tail: List[str] = []
    for arg in sys.argv[1:]:
        norm = arg.replace("\\", "/")
        if arg.endswith("eval_benchmarks.py"):
            continue
        if "/ipykernel" in norm:
            continue
        if arg == "-f" or (arg.startswith("-f") and len(arg) > 2):
            continue
        tail.append(arg)
    # Drop orphaned -f value if kernel.json slipped through
    cleaned: List[str] = []
    skip_next = False
    for arg in tail:
        if skip_next:
            skip_next = False
            continue
        if arg == "-f":
            skip_next = True
            continue
        cleaned.append(arg)
    tail = cleaned
    if tail:
        return tail
    env_cli = (os.environ.get("GP_BENCH_CLI") or "").strip()
    if env_cli:
        import shlex

        return shlex.split(env_cli)
    return ["--benchmark", "all", "--max_items", "100", "--seed", "42"]


def _kaggle_working_eval_script() -> str:
    return os.path.join(_work_dir(), "eval_benchmarks.py")


def _looks_like_eval_benchmarks_source(text: str) -> bool:
    if not text or len(text) < 50_000:
        return False
    markers = (
        "GP_BUNDLE_VERSION",
        "def main(",
        "_kaggle_reexec_if_notebook_kernel",
        "_ensure_eval_script_in_working",
    )
    return all(m in text for m in markers)


def _notebook_kernel_cell_source() -> str:
    """Recover full pasted-cell source from IPython/ipykernel linecache or input history."""
    import inspect
    import linecache

    best = ""
    try:
        for frame_info in inspect.stack():
            fn = frame_info.filename.replace("\\", "/")
            if (
                "ipykernel" in fn
                or fn.startswith("<ipython")
                or "ipython-input" in fn
            ):
                lines = linecache.getlines(frame_info.filename)
                if not lines:
                    continue
                text = "".join(lines)
                if len(text) > len(best) and _looks_like_eval_benchmarks_source(text):
                    best = text
    except Exception:
        pass

    if best:
        return best

    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except Exception:
        ip = None
    if ip is not None:
        try:
            hist = getattr(ip, "input_hist_parsed", None) or []
            for src in reversed(hist):
                if isinstance(src, str) and _looks_like_eval_benchmarks_source(src):
                    return src
        except Exception:
            pass
    return ""


def _materialize_pasted_eval_script_to_working(dest: str) -> bool:
    """Write a pasted notebook cell to /kaggle/working so !python and subprocess re-exec work."""
    if _is_eval_benchmarks_py(dest):
        return True
    if not _running_in_ipython_kernel():
        return False
    text = _notebook_kernel_cell_source()
    if not text:
        return False
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
            print(
            f"GP_BENCH: saved pasted notebook cell -> {dest!r} "
            f"({len(text):,} chars). Later cells may use "
            f"!python /kaggle/working/eval_benchmarks.py ...",
                flush=True,
            )
        return True
    except Exception as ex:
        print(f"GP_BENCH: could not save pasted script to working: {ex}", flush=True)
        return False


def _is_eval_benchmarks_py(path: str) -> bool:
    if not path or not str(path).endswith("eval_benchmarks.py"):
        return False
    if not os.path.isfile(path):
        return False
    try:
        return os.path.getsize(path) > 50_000
    except OSError:
        return False


def _find_eval_benchmarks_py() -> str:
    """Locate the full single-file script (notebook __file__ is often missing or ephemeral)."""
    seen: set[str] = set()
    candidates: List[str] = []

    def add(p: str) -> None:
        if not p:
            return
        ap = os.path.abspath(p)
        if ap in seen:
            return
        seen.add(ap)
        candidates.append(ap)

    add(_kaggle_working_eval_script())
    try:
        add(os.path.abspath(__file__))
    except NameError:
        pass
    main = sys.modules.get("__main__")
    mf = getattr(main, "__file__", None) if main is not None else None
    if mf and not str(mf).startswith("<"):
        add(str(mf))
    if sys.argv:
        add(sys.argv[0])
        for arg in sys.argv[1:]:
            if "eval_benchmarks.py" in arg:
                add(arg)
    sd = _script_dir()
    if sd:
        add(os.path.join(sd, "eval_benchmarks.py"))
    for root in _code_roots_for_import():
        add(os.path.join(root, "eval_benchmarks.py"))
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            p = os.path.join(kin, name, "eval_benchmarks.py")
            add(p)
            try:
                for sub in os.listdir(os.path.join(kin, name)):
                    add(os.path.join(kin, name, sub, "eval_benchmarks.py"))
            except OSError:
                pass
        try:
            import glob

            for p in glob.glob(os.path.join(kin, "**", "eval_benchmarks.py"), recursive=True):
                add(p)
        except OSError:
            pass
    try:
        import inspect

        add(inspect.getfile(_find_eval_benchmarks_py))
    except (TypeError, OSError):
        pass
    for extra in ("/kaggle/working/script.py",):
        add(extra)
    valid = [c for c in candidates if _is_eval_benchmarks_py(c)]
    if not valid:
        return ""
    valid.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return valid[0]


def _ensure_eval_script_in_working() -> str:
    """Stage eval_benchmarks.py under /kaggle/working so notebook subprocess re-exec works."""
    if not os.path.isdir("/kaggle"):
        return _find_eval_benchmarks_py()
    dest = _kaggle_working_eval_script()
    if _is_eval_benchmarks_py(dest):
        return dest
    src = _find_eval_benchmarks_py()
    if src and os.path.normpath(src) != os.path.normpath(dest):
        try:
            shutil.copy2(src, dest)
            print(f"GP_BENCH: staged eval_benchmarks.py for subprocess re-exec -> {dest!r}", flush=True)
        except Exception as ex:
            print(f"GP_BENCH: could not copy eval_benchmarks.py to working: {ex}", flush=True)
            return src
    if not _is_eval_benchmarks_py(dest):
        _materialize_pasted_eval_script_to_working(dest)
    return dest if _is_eval_benchmarks_py(dest) else (src if _is_eval_benchmarks_py(src) else "")


def _notebook_kernel_transformers_ok() -> bool:
    """True when Gemma-2 can import in this kernel (re-exec runs before faiss auto-pip in main)."""
    try:
        major, minor, _ver = _transformers_version_tuple()
        if (major, minor) < (4, 43):
            return False
        if not _gemma2_package_present():
            return False
            from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: F401

            _ = Gemma2ForCausalLM
        return True
    except Exception:
        return False


def _notebook_kernel_hf_ok(*, require_faiss: bool = True) -> bool:
    """True when this Jupyter kernel can load Gemma-2; faiss optional until after main() autopip."""
    if not _notebook_kernel_transformers_ok():
        return False
    if require_faiss:
        ok_faiss, _err = _probe_faiss_import()
        return bool(ok_faiss)
    return True


def _kaggle_reexec_if_notebook_kernel() -> None:
    """
    Notebook kernels keep a broken ``transformers`` import after pip (GenerationMixin, etc.).

    Re-launch this script with ``sys.executable`` so checks and model load use a clean interpreter.
    """
    if _env_truthy("GP_BENCH_SUBPROCESS"):
        return
    if not os.path.isdir("/kaggle"):
        return
    if _env_truthy("GP_BENCH_ALLOW_NOTEBOOK_RUN"):
        return
    if not _running_in_ipython_kernel():
        return

    _try_load_hf_token()

    import subprocess

    script = _ensure_eval_script_in_working()
    if not script or not _is_eval_benchmarks_py(script):
        if _notebook_kernel_transformers_ok():
            os.environ["GP_BENCH_ALLOW_NOTEBOOK_RUN"] = "1"
            faiss_ok, _ = _probe_faiss_import()
            extra = (
                " faiss-cpu will be installed during startup."
                if not faiss_ok
                else " faiss already available."
            )
            print(
                "GP_BENCH: Jupyter kernel ready (transformers OK) — running in-process."
                + extra
                + " Optional subprocess: copy eval_benchmarks.py to /kaggle/working/ and run "
                "!python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 100 --seed 42",
                flush=True,
            )
            return
        print(
            "GP_BENCH: cannot auto re-exec — paste the FULL eval_benchmarks.py in one cell "
            "(once to save /kaggle/working/eval_benchmarks.py), or attach it as a dataset, then:\n"
            "  !python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 100 --seed 42\n"
            "Or fix transformers in this kernel (Cell 1b + restart), then re-run.",
            flush=True,
        )
        return

    argv_tail = _kaggle_reexec_argv_tail()
    cmd = [sys.executable, script, *argv_tail]
    default_cli = ["--benchmark", "all", "--max_items", "100", "--seed", "42"]
    if argv_tail == default_cli and len(sys.argv) <= 1:
        print(
            "GP_BENCH: Jupyter stripped kernel -f args; using default CLI: "
            f"{' '.join(argv_tail)} "
            "(override with GP_BENCH_CLI or !python ... --benchmark all --seed 42)",
            flush=True,
        )
    print(
        "\nKaggle: notebook kernel detected — re-launching in a fresh Python subprocess "
        "(avoids broken in-kernel transformers after pip):\n"
        f"  {' '.join(cmd)}\n",
        flush=True,
    )
    env = {**os.environ, "GP_BENCH_SUBPROCESS": "1"}
    proc = subprocess.run(cmd, env=env)
    raise SystemExit(proc.returncode)


def _warn_notebook_kernel_on_kaggle() -> None:
    """No-op when auto re-exec handles notebook kernels; kept for GP_BENCH_ALLOW_NOTEBOOK_RUN."""
    if not os.path.isdir("/kaggle") or not _running_in_ipython_kernel():
        return
    if _env_truthy("GP_BENCH_ALLOW_NOTEBOOK_RUN"):
        print(
            "GP_BENCH: GP_BENCH_ALLOW_NOTEBOOK_RUN=1 — continuing in notebook kernel.",
            flush=True,
        )


def _kaggle_prepare_environment() -> None:
    """Unpack helpers to /kaggle/working and extend sys.path before imports that need the runner."""
    if not os.path.isdir("/kaggle"):
        return
    _quiet_hf_datasets()
    _ensure_eval_script_in_working()
    work = os.path.abspath(_work_dir())
    for p in (work, "/kaggle/working", os.getcwd()):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here and here not in sys.path:
            sys.path.insert(0, here)
    except NameError:
        pass
    try:
        _setup_runner_path()
    except Exception as ex:
        print(f"GP_BENCH: helper setup warning ({ex})", flush=True)


def _transformers_version_tuple() -> Tuple[int, int, str]:
    ver = "0"
    try:
        import importlib.metadata as im

        ver = str(im.version("transformers"))
    except Exception:
        try:
            import transformers

            ver = str(getattr(transformers, "__version__", "0"))
        except Exception:
            pass
    parts = ver.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor, ver


def _gemma2_package_present() -> bool:
    try:
        import transformers

        root = os.path.dirname(getattr(transformers, "__file__", "") or "")
        path = os.path.join(root, "models", "gemma2", "modeling_gemma2.py")
        return bool(root) and os.path.isfile(path)
    except Exception:
        return False


def _probe_hf_imports_subprocess() -> str:
    """Fresh-interpreter import test (safe after Cell 1b + kernel restart)."""
    import subprocess

    code = (
        "import transformers\n"
        "from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM\n"
        "import huggingface_hub\n"
        "print(transformers.__version__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "import probe failed").strip()
        raise ImportError(err)
    return (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""


def _probe_faiss_import() -> Tuple[bool, str]:
    try:
        import faiss  # type: ignore  # noqa: F401

        return True, ""
    except Exception as ex:
        return False, str(ex)


def _ensure_faiss_cpu_installed(*, try_autopip: bool = True) -> Tuple[bool, str]:
    """
    Dense RAG requires ``faiss-cpu``. On Kaggle, auto-pip only ``faiss-cpu`` (safe in a fresh
    ``!python`` process; does not touch transformers like the full HF stack auto-pip).
    """
    ok, err = _probe_faiss_import()
    if ok:
        if os.path.isdir("/kaggle"):
            import faiss  # type: ignore

            print(f"Kaggle: faiss {getattr(faiss, '__version__', '?')} OK (dense RAG).", flush=True)
        return True, ""
    if not try_autopip or _env_truthy("GP_BENCH_NO_AUTO_PIP"):
        return False, err or "No module named 'faiss'"
    import subprocess
    import sys

    print(
        "GP_BENCH: installing faiss-cpu (required for FAISS retrieval; lexical BM25 cannot "
        "differentiate recall@1 vs recall@k)...",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"],
        check=False,
        timeout=300,
    )
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:
        pass
    ok, err = _probe_faiss_import()
    if ok:
        import faiss  # type: ignore

        print(f"Kaggle: faiss {getattr(faiss, '__version__', '?')} OK after auto-pip.", flush=True)
        return True, ""
    return False, err or "No module named 'faiss'"


def _require_faiss_for_rag_run(
    *,
    allow_lexical: bool,
    rag_use_mock: bool,
    mock: bool,
    hf_router: bool,
    try_autopip: bool = True,
) -> None:
    """Abort before 3000+ lexical-only RAG rows when dense retrieval is required."""
    if mock or rag_use_mock or hf_router or allow_lexical:
        return
    if not any(use_rag for _name, _key, use_rag in CONFIGS):
        return
    ok, err = _ensure_faiss_cpu_installed(try_autopip=try_autopip)
    if ok:
        return
    if os.path.isdir("/kaggle") and not _kaggle_strict_checks():
        print(
            f"RAG WARNING: faiss-cpu missing ({err}). Dense retrieval disabled; "
            "pass --allow_lexical_rag only for debugging. Install faiss-cpu in Cell 1b and restart.",
            flush=True,
        )
        return
    cell1b = (
        "  !pip install -q 'transformers>=4.43.0,<5' 'huggingface-hub>=0.23.0' "
        "'accelerate>=0.26.0' 'sentence-transformers>=2.2.2' 'safetensors>=0.4.0' "
        "faiss-cpu 'bitsandbytes>=0.46.1' --upgrade\n"
        "  # Kernel → Restart Session, then: python -c \"import faiss; print(faiss.__version__)\""
    )
    raise RuntimeError(
        "faiss-cpu is not installed — all RAG configs would use lexical fallback only.\n"
        "Lexical BM25 cannot fix differentiated recall@k (gold chunk is always rank 1 or absent) "
        "and can inject irrelevant context (Fault 3 / Fault 6).\n\n"
        f"  Import error: {err}\n\n"
        f"Cell 1b after restart:\n{cell1b}\n\n"
        "Or pass --allow_lexical_rag to run anyway (not valid for RAG ablation results)."
    )


def _maybe_autopip_hf_stack_on_kaggle() -> bool:
    """
    In-process pip + re-import always corrupts transformers on Kaggle.

    Opt-in only via GP_BENCH_ALLOW_AUTO_PIP=1; even then we only print the pip line and exit.
    """
    if _env_truthy("GP_BENCH_NO_AUTO_PIP"):
        return False
    if not _env_truthy("GP_BENCH_ALLOW_AUTO_PIP"):
        return False
    if not os.path.isdir("/kaggle"):
        return False
    import subprocess

    print(
        "GP_BENCH: auto-pip is disabled by default on Kaggle (it breaks transformers in-session). "
        f"Run manually, then restart:\n  !{_kaggle_hf_stack_pip_line()}",
        flush=True,
    )
    return False


def _require_transformers_hf_stack(*, try_autopip: bool = True) -> None:
    """
    Gemma-2 and sentence-transformers need a recent transformers (>=4.43).

    Never pip-install transformers inside this process; use Cell 1b + kernel restart + ``!python``.
    """
    hf_line = _kaggle_hf_stack_pip_line()

    def _check_imports() -> str:
        major, minor, ver = _transformers_version_tuple()
        if (major, minor) < (4, 43):
            raise ImportError(f"transformers {ver} is too old for google/gemma-2-2b-it (need >=4.43)")
        if not _gemma2_package_present():
            raise ImportError(f"transformers {ver} lacks Gemma-2 files (models/gemma2/modeling_gemma2.py)")
        import transformers
        from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: F401

        _ = Gemma2ForCausalLM
        import huggingface_hub  # noqa: F401

        return ver

    try:
        ver = _check_imports()
    except Exception as ex:
        probe_hint = ""
        probe_ok = False
        if os.path.isdir("/kaggle"):
            try:
                probe_ver = _probe_hf_imports_subprocess()
                probe_ok = True
                probe_hint = (
                    f"\n\nA fresh ``python -c`` probe succeeded (transformers {probe_ver}), "
                    "but this Jupyter kernel failed above. Run benchmarks in a subprocess:\n"
                    "  !python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 100 --seed 42\n"
                    "(not %%run / Run Cell on eval_benchmarks.py). Then Kernel → Restart after Cell 1b."
                )
            except Exception:
                pass
            if probe_ok and _running_in_ipython_kernel() and not _env_truthy("GP_BENCH_SUBPROCESS"):
                _kaggle_reexec_if_notebook_kernel()
                raise RuntimeError(
                    "Transformers works in a fresh Python subprocess but not in this notebook kernel.\n"
                    f"  Kernel error: {ex}{probe_hint}"
                )
        if try_autopip:
            _maybe_autopip_hf_stack_on_kaggle()
        raise RuntimeError(
            "Hugging Face stack is broken or too old for Gemma-2 / RAG embeddings.\n"
            f"  {ex}{probe_hint}\n\n"
            "On Kaggle:\n"
            "  1. Kernel → Restart Session\n"
            "  2. Cell 1 (numpy/scipy/pandas) → restart\n"
            "  3. Cell 1b (HF stack) → restart again\n"
            "  4. Cell 2 verify imports\n"
            "  5. Cell 3 ONLY: !python eval_benchmarks.py ...  (not %%run)\n\n"
            f"  !{hf_line}\n\n"
            "Verify in Cell 2:\n"
            "  from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM\n"
            "  import transformers; print(transformers.__version__)\n"
            "Do not pip install pandas/numpy alone without Cell 1b + restart afterward."
        ) from ex

    if os.path.isdir("/kaggle"):
        print(f"Kaggle: transformers {ver} OK (Gemma-2 / RAG embed).", flush=True)


def _require_numpy_scipy_stack() -> None:
    """
    ``datasets`` pulls in ``scipy`` when building some Hub loaders (e.g. MedQA).

    A broken numpy↔scipy pair (common after ``pip install pandas --force-reinstall`` alone)
    fails with ``ImportError: cannot import name '_center' from numpy._core.umath``.
    """
    try:
        import numpy as np

        np_ver = str(getattr(np, "__version__", ""))
        # Same import chain as ``datasets.streaming`` → ``scipy.io.loadmat``.
        import scipy.sparse  # noqa: F401
        from scipy.io.matlab import loadmat  # noqa: F401

        _ = loadmat  # used
        if os.path.isdir("/kaggle"):
            import scipy

            print(
                f"Kaggle: numpy {np_ver} + scipy {scipy.__version__} OK.",
                flush=True,
            )
    except Exception as ex:
        pip_line = _kaggle_numpy_scipy_pip_line()
        alt = (
            "pip install -q 'numpy>=2.0,<2.2' 'scipy>=1.14.1' "
            f"'{_KAGGLE_PANDAS_SPEC}' 'datasets>=2.14' pyarrow --force-reinstall"
        )
        raise RuntimeError(
            "numpy and scipy are incompatible (MedQA / Hugging Face datasets will fail).\n"
            f"  {ex}\n\n"
            "On Kaggle: Kernel → Restart Session, then run ONE cell (before eval_benchmarks):\n"
            f"  !{_kaggle_numpy_scipy_pip_line()}\n"
            "Verify:\n"
            "  import numpy, scipy; print(numpy.__version__, scipy.__version__)\n"
            f"If that still fails, try:\n  !{alt}\n"
            "Do not pip install pandas alone without numpy+scipy in the same command."
        ) from ex


def _shadowing_module_paths(mod_name: str) -> List[str]:
    roots: List[str] = []
    for r in (os.getcwd(), _work_dir(), "/kaggle/working"):
        if r and r not in roots:
            roots.append(r)
    try:
        sd = _script_dir()
        if sd and sd not in roots:
            roots.append(sd)
    except Exception:
        pass
    return [os.path.join(r, f"{mod_name}.py") for r in roots]


def _import_load_dataset() -> Any:
    """
    Import Hugging Face ``load_dataset`` safely.

    On Kaggle notebooks, a local ``pandas.py`` or a half-initialized ``pandas`` in
    ``sys.modules`` causes ``AttributeError: partially initialized module 'pandas'``.
    """
    global _LOAD_DATASET_FN
    if _LOAD_DATASET_FN is not None:
        return _LOAD_DATASET_FN

    for mod in ("pandas", "datasets"):
        for p in _shadowing_module_paths(mod):
            if os.path.isfile(p):
                raise RuntimeError(
                    f"Found {p!r}, which shadows the installed '{mod}' package. "
                    f"Rename or remove that file, then Kernel → Restart Session."
                )

    def _purge(mod: str) -> None:
        for key in list(sys.modules):
            if key == mod or key.startswith(mod + "."):
                del sys.modules[key]

    for mod in ("pandas", "datasets"):
        m = sys.modules.get(mod)
        if m is None:
            continue
        ok = (mod == "pandas" and hasattr(m, "core")) or (
            mod == "datasets" and hasattr(m, "load_dataset")
        )
        if not ok:
            _purge(mod)

    # Import before /kaggle/working is prepended to sys.path (local pandas.py shadows site-packages).
    shadow_roots = [
        r
        for r in (os.getcwd(), _work_dir(), "/kaggle/working")
        if r and os.path.isdir(r)
    ]
    saved_path = list(sys.path)
    try:
        for r in shadow_roots:
            while r in sys.path:
                sys.path.remove(r)
        import pandas as pd  # noqa: F401

        if not hasattr(pd, "core"):
            raise ImportError("pandas import incomplete")
        from datasets import load_dataset

        _quiet_hf_datasets()
    except AttributeError as ex:
        _purge("pandas")
        _purge("datasets")
        raise ImportError(
            "pandas/datasets import failed (often a local pandas.py on sys.path, or a stale "
            "kernel). Remove /kaggle/working/pandas.py if present, Kernel → Restart Session, "
            f"then: pip install -q '{_KAGGLE_PANDAS_SPEC}' 'datasets>=2.14' pyarrow "
            "(do not install pandas 3.x on Kaggle)."
        ) from ex
    except Exception as ex:
        raise ImportError(
            "pandas failed to import (required by `datasets`). On Kaggle: remove any "
            "local pandas.py, then Kernel → Restart Session and "
            f"pip install -q '{_KAGGLE_PANDAS_SPEC}' 'datasets>=2.14' pyarrow."
        ) from ex
    finally:
        sys.path[:] = saved_path

    _require_pandas2_on_kaggle()
    _require_numpy_scipy_stack()
    _LOAD_DATASET_FN = load_dataset
    return load_dataset


def summarize_rag_in_predictions(predictions_json: str) -> Dict[str, Any]:
    """
    Quick post-run check: RAG rows should use ``faiss`` or ``lexical``, not legacy ``mock``.

    Example (notebook)::

        from eval_benchmarks import summarize_rag_in_predictions
        summarize_rag_in_predictions("/kaggle/working/benchmark_results_all_predictions.json")
    """
    path = os.path.abspath(predictions_json)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = list(payload.get("rows") or [])
    rag_rows = [r for r in rows if r.get("rag_flag")]
    by_source: Dict[str, int] = {}
    mock_snippet = 0
    rows_with_hits = 0
    sample_sources: List[str] = []
    pubmed_rag: Dict[str, List[float]] = {}
    for r in rag_rows:
        src = str(r.get("rag_source") or "missing")
        by_source[src] = by_source.get(src, 0) + 1
        ctx = str(r.get("retrieved_context") or "")
        hits = r.get("rag_hits")
        if isinstance(hits, list) and hits:
            rows_with_hits += 1
            for h in hits[:3]:
                if isinstance(h, dict):
                    s = str(h.get("source") or "")
                    if s and s not in sample_sources:
                        sample_sources.append(s)
        if "metformin" in ctx.lower() and "FDA Guideline" in ctx:
            mock_snippet += 1
        if str(r.get("benchmark")) == "pubmedqa":
            mname = str(r.get("model_name") or "")
            hit = r.get("label_correct") is True or r.get("label_correct") == 1.0
            pubmed_rag.setdefault(mname, []).append(1.0 if hit else 0.0)
    pubmed_acc = {
        m: (sum(v) / len(v) if v else None) for m, v in sorted(pubmed_rag.items())
    }
    retr_by_model: Dict[str, Dict[str, Any]] = {}
    for r in rag_rows:
        if not r.get("retrieval_evaluable"):
            continue
        mname = str(r.get("model_name") or "")
        retr_by_model.setdefault(mname, {"mrr": [], "recall": {k: [] for k in _rag_recall_ks()}})
        if r.get("mrr") == r.get("mrr"):
            retr_by_model[mname]["mrr"].append(float(r["mrr"]))
        for k in _rag_recall_ks():
            key = f"recall_at_{k}"
            if key in r and r[key] == r[key]:
                retr_by_model[mname]["recall"][k].append(float(r[key]))
    retrieval_summary: Dict[str, Any] = {}
    for mname, buckets in sorted(retr_by_model.items()):
        entry: Dict[str, Any] = {}
        if buckets["mrr"]:
            entry["mrr_mean"] = sum(buckets["mrr"]) / len(buckets["mrr"])
        for k, vals in buckets["recall"].items():
            if vals:
                entry[f"recall_at_{k}_mean"] = sum(vals) / len(vals)
        retrieval_summary[mname] = entry
    retr_na = False
    if _external_corpus_retrieval_metrics_na():
        if not retrieval_summary or all(
            _recall_means_all_zero(entry, _rag_recall_ks()) for entry in retrieval_summary.values()
        ):
            retr_na = True
    out: Dict[str, Any] = {
        "predictions_path": path,
        "n_rag_rows": len(rag_rows),
        "rag_source_counts": by_source,
        "rows_with_rag_hits": rows_with_hits,
        "sample_corpus_sources": sample_sources[:12],
        "rows_with_diabetes_mock_snippet": mock_snippet,
        "pubmedqa_label_accuracy_rag_configs": pubmed_acc,
        "retrieval_metrics_by_rag_config": retrieval_summary,
    }
    if retr_na:
        out["retrieval_metrics_na"] = True
        out["retrieval_na_reason"] = "external_non_leaking_corpus_no_pubmedqa_gold"
    print(json.dumps(out, indent=2), flush=True)
    if retr_na:
        print(_RETRIEVAL_NA_SUMMARY_MSG, flush=True)
    if by_source.get("mock"):
        print(
            "WARNING: legacy mock rag_source — re-run with updated real_model_runner.py.",
            flush=True,
        )
    elif mock_snippet:
        print(
            "WARNING: diabetes placeholder snippets in retrieved_context — check corpus / index.",
            flush=True,
        )
    elif by_source.get("lexical") and not by_source.get("faiss"):
        print(
            "WARNING: all RAG rows used lexical fallback — FAISS was not loaded. "
            "Set --rag_index_dir, install faiss-cpu + sentence-transformers, and "
            "RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2.",
            flush=True,
        )
    elif by_source.get("faiss") or by_source.get("lexical"):
        print(
            "OK: RAG rows use real corpus retrieval (faiss and/or lexical); no placeholder snippets.",
            flush=True,
        )
    return out


def recompute_pubmed_label_accuracy(predictions_json: str) -> Dict[str, Any]:
    """
    Re-score PubMedQA label accuracy from ``raw_response`` with case-normalised parsing
    (no re-inference). Useful after fixing ``_parse_pubmed_model_answer``.
    """
    path = os.path.abspath(predictions_json)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = [r for r in (payload.get("rows") or []) if str(r.get("benchmark")) == "pubmedqa"]
    by_model: Dict[str, List[float]] = {}
    for r in rows:
        ref = str(r.get("reference_answer") or "").strip().lower()
        parsed = _parse_pubmed_model_answer(str(r.get("raw_response") or ""))
        hit = 1.0 if ref and parsed and parsed.lower() == ref else 0.0
        mname = str(r.get("model_name") or "")
        by_model.setdefault(mname, []).append(hit)
    out = {
        k: (sum(v) / len(v) if v else None) for k, v in sorted(by_model.items())
    }
    print(json.dumps({"pubmedqa_label_accuracy_fixed": out, "n_rows": len(rows)}, indent=2), flush=True)
    return out


def _load_medqa(split: str = "validation") -> List[Dict[str, Any]]:
    load_dataset = _import_load_dataset()

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


def _medmcqa_answer_letter(row: Dict[str, Any]) -> str:
    """Map MedMCQA ``cop`` (0–3) to A–D; audit m8."""
    cop = row.get("cop")
    if cop is not None and cop != "":
        try:
            i = int(cop)
            if 0 <= i <= 3:
                return chr(ord("A") + i)
        except (TypeError, ValueError):
            pass
    return _normalize_mcq_gold(str(row.get("answer") or ""), 4)


def _load_medmcqa(split: str = "validation") -> List[Dict[str, Any]]:
    load_dataset = _import_load_dataset()

    ds = load_dataset("medmcqa", split=split)
    items: List[Dict[str, Any]] = []
    for row in ds:
        choices = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
        items.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question") or "").strip(),
                "choices": choices,
                "answer": _medmcqa_answer_letter(row),
            }
        )
    return items


# HF ``pqa_labeled`` only exposes ``train`` (1000 QA-RL examples with yes/no/maybe labels).
PUBMEDQA_LABELED_SPLIT = "train"
PUBMEDQA_HOLDOUT_SEED_DEFAULT = 42
PUBMEDQA_HOLDOUT_FRAC_DEFAULT = 0.2

_PUBMEDQA_SPLIT_STATE: Dict[str, Any] = {}


def _pubmedqa_holdout_seed() -> int:
    raw = os.environ.get("PUBMEDQA_HOLDOUT_SEED", str(PUBMEDQA_HOLDOUT_SEED_DEFAULT)).strip()
    try:
        return int(raw)
    except ValueError:
        return PUBMEDQA_HOLDOUT_SEED_DEFAULT


def _pubmedqa_holdout_frac() -> float:
    raw = os.environ.get("PUBMEDQA_HOLDOUT_FRAC", str(PUBMEDQA_HOLDOUT_FRAC_DEFAULT)).strip()
    try:
        f = float(raw)
    except ValueError:
        f = PUBMEDQA_HOLDOUT_FRAC_DEFAULT
    return max(0.05, min(0.5, f))


def _pubmedqa_eval_split_descriptor() -> str:
    pct = int(round(_pubmedqa_holdout_frac() * 100))
    return f"pqa_labeled_holdout_{pct}pct_seed{_pubmedqa_holdout_seed()}"


def _split_pubmedqa_corpus_eval(
    items: List[Dict[str, Any]],
    *,
    seed: Optional[int] = None,
    holdout_frac: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str], Set[str]]:
    """Hold out a fixed fraction of PMIDs for eval; remainder builds the gold retrieval corpus (C1)."""
    seed = PUBMEDQA_HOLDOUT_SEED_DEFAULT if seed is None else int(seed)
    holdout_frac = PUBMEDQA_HOLDOUT_FRAC_DEFAULT if holdout_frac is None else float(holdout_frac)
    by_id: Dict[str, Dict[str, Any]] = {}
    for it in items:
        pid = str(it.get("id") or "").strip()
        if pid:
            by_id[pid] = it
    ids = sorted(by_id.keys())
    rng = random.Random(seed)
    order = list(ids)
    rng.shuffle(order)
    n_hold = max(1, int(round(len(order) * holdout_frac)))
    holdout_ids = set(order[:n_hold])
    corpus_ids = set(order[n_hold:])
    eval_items = [by_id[i] for i in order if i in holdout_ids]
    corpus_items = [by_id[i] for i in order if i in corpus_ids]
    return eval_items, corpus_items, holdout_ids, corpus_ids


def _ensure_pubmedqa_split_loaded() -> Dict[str, Any]:
    if _PUBMEDQA_SPLIT_STATE.get("loaded"):
        return _PUBMEDQA_SPLIT_STATE
    all_items = _load_pubmedqa_labeled_rows()
    eval_items, corpus_items, holdout_ids, corpus_ids = _split_pubmedqa_corpus_eval(all_items)
    _PUBMEDQA_SPLIT_STATE.update(
        {
            "loaded": True,
            "n_all": len(all_items),
            "n_eval": len(eval_items),
            "n_corpus": len(corpus_items),
            "holdout_pubids": holdout_ids,
            "corpus_pubids": corpus_ids,
            "eval_items": eval_items,
            "corpus_items": corpus_items,
            "eval_split_descriptor": _pubmedqa_eval_split_descriptor(),
            "hf_config": "pubmed_qa/pqa_labeled",
            "hf_split_loaded": PUBMEDQA_LABELED_SPLIT,
        }
    )
    print(
        f"\nPubMedQA split: eval holdout n={len(eval_items)} ({_pubmedqa_holdout_frac():.0%} of "
        f"{len(all_items)}), gold corpus n={len(corpus_items)} "
        f"({_pubmedqa_eval_split_descriptor()}). Not an official PubMedQA test split (C3).\n",
        flush=True,
    )
    return _PUBMEDQA_SPLIT_STATE


def _pubmedqa_eval_pubids() -> Set[str]:
    return set(_ensure_pubmedqa_split_loaded().get("holdout_pubids") or set())


def _pubmedqa_pubids_in_gold_chunks(gold_dir: str) -> Set[str]:
    """PMIDs with a ``pubmedqa_<pmid>`` (or ``_long``) row in gold ``chunks.jsonl``."""
    cp = os.path.join(os.path.abspath(gold_dir), "chunks.jsonl")
    if not os.path.isfile(cp):
        return set()
    found: Set[str] = set()
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = str(obj.get("source") or "")
            if not src.startswith("pubmedqa_"):
                continue
            body = src[len("pubmedqa_") :]
            pubid = body.split("_", 1)[0]
            if pubid:
                found.add(pubid)
    return found


def _gold_index_missing_eval_pubids(gold_dir: str, eval_pubids: Set[str]) -> List[str]:
    """Eval holdout PMIDs absent from gold index — recall@k on those items will stay 0."""
    if not eval_pubids or not gold_dir:
        return sorted(eval_pubids) if eval_pubids else []
    in_index = _pubmedqa_pubids_in_gold_chunks(gold_dir)
    return sorted(pid for pid in eval_pubids if pid not in in_index)


def _write_pubmedqa_gold_manifest(out_dir: str, state: Dict[str, Any]) -> str:
    path = os.path.join(out_dir, "pubmedqa_gold_manifest.json")
    holdout = sorted(state.get("holdout_pubids") or [])
    payload = {
        "version": GP_BUNDLE_VERSION,
        "eval_split_descriptor": state.get("eval_split_descriptor"),
        "hf_config": state.get("hf_config"),
        "hf_split_loaded": state.get("hf_split_loaded"),
        "n_all_labeled": state.get("n_all"),
        "n_eval_holdout": state.get("n_eval"),
        "n_corpus_for_index": state.get("n_corpus"),
        "holdout_seed": _pubmedqa_holdout_seed(),
        "holdout_frac": _pubmedqa_holdout_frac(),
        "holdout_pubids_sample": holdout[:20],
        "n_holdout_pubids": len(holdout),
        "retrieval_query_default": "question_only",
        "corpus_pubids_excluded_from_eval_holdout": True,
        "eval_pubids_in_gold_index_for_recall_eval": True,
        "task_rag_uses_primary_external_index": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _build_rag_experiment_manifest(benchmarks: List[str]) -> Dict[str, Any]:
    """Explicit dual-corpus reporting (audit C4)."""
    primary = os.environ.get("RAG_INDEX_DIR", "").strip()
    gold = _pubmedqa_rag_index_dir()
    split = _ensure_pubmedqa_split_loaded() if "pubmedqa" in benchmarks else {}
    rq = (
        "question_plus_abstract"
        if _pubmedqa_retrieval_query_includes_abstract()
        else "question_only"
    )
    dual = bool(gold and primary and "pubmedqa" in benchmarks)
    out: Dict[str, Any] = {
        "mcq_rag_corpus": "primary_external",
        "mcq_rag_index_dir": primary or None,
        "pubmedqa_eval_split": split.get("eval_split_descriptor", "n/a"),
        "pubmedqa_eval_n_holdout": split.get("n_eval"),
        "pubmedqa_corpus_n_for_gold_index": split.get("n_corpus"),
        "pubmedqa_retrieval_query_mode": rq,
        "pubmedqa_task_rag_index_dir": primary or None,
        "pubmedqa_retrieval_eval_index_dir": gold or None,
        "pubmedqa_rag_index_role": "gold_for_recall_eval_primary_for_generation" if dual else "primary_or_none",
        "pubmedqa_rag_index_dir": gold or primary or None,
        "dual_corpus_active": dual,
        "mcq_rag_interpretation": (
            "MCQ RAG uses the external ~25k corpus (medquad/pubmed_*). Low task accuracy with RAG "
            "often reflects corpus mismatch, not retriever failure. PubMedQA recall@k requires the "
            "gold holdout index (dual_corpus_active=true)."
        ),
        "reporting_requirement": (
            "Report MCQ task metrics (external RAG) separately from PubMedQA label accuracy "
            "and PubMedQA retrieval metrics (gold holdout index). Do not merge into one RAG headline."
        ),
    }
    if dual:
        print(
            "\nRAG EXPERIMENT (dual corpus): MCQ → primary external index; "
            "PubMedQA RAG/recall@k → gold holdout index. See rag_experiment in results JSON.\n",
            flush=True,
        )
    return out


def _pubmedqa_generative_metrics_enabled() -> bool:
    """One-word PubMedQA outputs: BLEU/ROUGE/BERTScore vs long_answer are invalid (audit C5)."""
    return _env_truthy("GP_PUBMEDQA_ENABLE_GENERATIVE_METRICS")


def _mmlu_med_split(cli_split: str = "") -> str:
    """Default ``dev`` to reduce test-set contamination claims (audit C6)."""
    s = (cli_split or os.environ.get("MMLU_MED_SPLIT", "dev")).strip().lower()
    if s in ("dev", "validation", "test", "train"):
        return s
    return "dev"


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
    return _normalize_pubmed_label(str(raw).strip().lower())


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


def _pubmedqa_user_prompt(question: str, context: str, model_key: str = "") -> str:
    """Real PubMedQA-style prompt: abstract context + question (no fabricated demo text)."""
    q = (question or "").strip()
    ctx = (context or "").strip()
    if model_key == "slm":
        suffix = (
            "You MUST choose exactly one word: yes, no, or maybe. "
            "If studies show no effect, negative findings, or the abstract contradicts the claim, answer: no. "
            '"no" is a valid and expected answer — do not default to maybe unless the abstract is truly inconclusive. '
            "Use maybe only when evidence is insufficient or clearly mixed. "
            "Do not explain.\n\nFinal answer (yes, no, or maybe):"
        )
    else:
        suffix = (
            "Answer with exactly one word: yes, no, or maybe. "
            "If the abstract clearly contradicts the question, answer: no. "
            "If the evidence is insufficient or mixed, answer: maybe. "
            "Do not explain.\n\nFinal answer (yes, no, or maybe):"
        )
    if ctx:
        return (
            "Read the biomedical abstract below, then answer the question.\n\n"
            f"Abstract:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"{suffix}"
        )
    return f"{q}\n\n{suffix}"


def _pubmedqa_max_new_tokens(model_key: str = "", effective_rag: bool = False) -> int:
    if model_key == "llm" and effective_rag:
        default = "40"
        cap = 64
    elif model_key == "slm":
        default = "32"
        cap = 48
    else:
        default = "28"
        cap = 48
    raw = os.environ.get("PUBMEDQA_MAX_NEW_TOKENS", default).strip()
    try:
        return max(12, min(cap, int(raw)))
    except ValueError:
        return int(default)


def _load_pubmedqa_labeled_rows(split: str = PUBMEDQA_LABELED_SPLIT) -> List[Dict[str, Any]]:
    load_dataset = _import_load_dataset()

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


def _load_pubmedqa(split: str = PUBMEDQA_LABELED_SPLIT) -> List[Dict[str, Any]]:
    """Eval items only: fixed holdout from ``pqa_labeled`` (not official PubMedQA test)."""
    _ = split
    return list(_ensure_pubmedqa_split_loaded()["eval_items"])


def _load_pubmedqa_corpus_rows() -> List[Dict[str, Any]]:
    """Rows used to build gold ``chunks.jsonl`` (excludes eval holdout PMIDs)."""
    return list(_ensure_pubmedqa_split_loaded()["corpus_items"])


# Medical-related MMLU subjects (``cais/mmlu``); default eval split is ``dev`` (see ``_mmlu_med_split``).
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


def _load_mmlu_med(split: Optional[str] = None) -> List[Dict[str, Any]]:
    """MMLU medical slice: multiple subjects from ``cais/mmlu`` (4-way MCQ)."""
    load_dataset = _import_load_dataset()
    use_split = _mmlu_med_split(split or "")

    items: List[Dict[str, Any]] = []
    for subj in MMLU_MED_SUBJECTS:
        ds = load_dataset("cais/mmlu", subj, split=use_split)
        for i, row in enumerate(ds):
            choices = [str(c) for c in row["choices"]]
            items.append(
                {
                    "id": f"{subj}_{i}",
                    "subject": subj,
                    "question": str(row.get("question", "")).strip(),
                    "choices": choices,
                    "answer": _mmlu_answer_letter(row),
                }
            )
    return items


def _bootstrap_ci(values: List[float], alpha: float = 0.05, iters: int = 2000) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = _bootstrap_rng()
    n = len(values)
    mean = sum(values) / n
    samples: List[float] = []
    for _ in range(iters):
        samples.append(sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n)
    samples.sort()
    lo = int((alpha / 2) * iters)
    hi = int((1 - alpha / 2) * iters)
    return mean, samples[lo], samples[min(hi, iters - 1)]


def _binary_mean_std(bits: List[float]) -> Tuple[float, float]:
    if not bits:
        return float("nan"), float("nan")
    mean = statistics.mean(bits)
    if len(bits) < 2:
        return mean, 0.0
    return mean, statistics.stdev(bits)


def _accuracy(preds: List[Any], golds: List[Any]) -> Tuple[float, float, float]:
    return _bootstrap_ci([1.0 if p == g else 0.0 for p, g in zip(preds, golds)])


def _token_f1(pred: str, ref: str) -> float:
    from eval_quality_metrics import compute_f1

    return compute_f1(pred, ref)["f1"]


def _mcq_choice_text(choices: Sequence[str], letter: str) -> str:
    """Full option text for the given letter (A–D); falls back to the letter if invalid."""
    let = (letter or "").strip().upper()
    if len(let) == 1 and "A" <= let <= "Z":
        idx = ord(let) - ord("A")
        if 0 <= idx < len(choices):
            return str(choices[idx]).strip()
    return let


def _pubmedqa_metric_reference(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (label yes|no|maybe, reference text for token/ROUGE/BERTScore).

    Token-level F1 vs the one-word label is degenerate (0/1 only). Use ``long_answer``.
    """
    label = str(item.get("answer") or "").strip().lower()
    long_a = str(item.get("long_answer") or "").strip()
    ref_text = long_a if long_a else label
    return label, ref_text


def _rouge_l(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return _bootstrap_ci(scores)


def _merge_metric_ci(
    out: Dict[str, Any], prefix: str, stats: Tuple[float, float, float]
) -> None:
    mean, lo, hi = stats
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_ci_lower"] = lo
    out[f"{prefix}_ci_upper"] = hi


def _bleu(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    try:
        import sacrebleu
    except ImportError:
        print("BLEU skipped: pip install sacrebleu", flush=True)
        return (float("nan"),) * 3
    scores = [
        sacrebleu.sentence_bleu(str(p), [str(r)]).score / 100.0 for p, r in zip(preds, refs)
    ]
    return _bootstrap_ci(scores)


def _ensure_nltk_meteor_data() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _meteor(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        print("METEOR skipped: pip install nltk", flush=True)
        return (float("nan"),) * 3
    try:
        _ensure_nltk_meteor_data()
        scores: List[float] = []
        for p, r in zip(preds, refs):
            ref_toks = word_tokenize(str(r).lower())
            pred_toks = word_tokenize(str(p).lower())
            if not ref_toks and not pred_toks:
                scores.append(1.0)
            elif not ref_toks or not pred_toks:
                scores.append(0.0)
            else:
                scores.append(float(meteor_score([ref_toks], pred_toks)))
        return _bootstrap_ci(scores)
    except Exception as ex:
        print(f"METEOR skipped: {ex}", flush=True)
        return (float("nan"),) * 3


def _bertscore_model_type() -> str:
    return (
        os.environ.get("GP_BERTSCORE_MODEL", "").strip()
        or "roberta-large-L17"
    )


def _bertscore_f1(preds: List[str], refs: List[str]) -> Tuple[float, float, float, str]:
    model_type = _bertscore_model_type()
    if not preds:
        return (float("nan"),) * 3 + (model_type,)
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("BERTScore skipped: pip install bert-score", flush=True)
        return (float("nan"),) * 3 + (model_type,)
    try:
        _, _, f1 = bert_score_fn(
            [str(p) for p in preds],
            [str(r) for r in refs],
            lang="en",
            model_type=model_type,
            verbose=False,
            batch_size=min(32, max(1, len(preds))),
        )
        m, lo, hi = _bootstrap_ci([float(x) for x in f1.tolist()])
        return m, lo, hi, model_type
    except Exception as ex:
        print(f"BERTScore skipped: {ex}", flush=True)
        return (float("nan"),) * 3 + (model_type,)


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
    tail_m = re.search(
        r"FINAL\s+ANSWER\s+LETTER\s*:?\s*([A-Z])",
        u,
        flags=re.IGNORECASE,
    )
    if tail_m and "A" <= tail_m.group(1).upper() <= last:
        return tail_m.group(1).upper()
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


def build_fixed_chunks(
    input_jsonl: str,
    output_jsonl: str,
    chunk_words: int = 300,
    overlap_words: int = 54,
) -> int:
    """
    Split long chunks into overlapping windows (~256–384 tokens; default ~300 words, 18% overlap).

    The first sub-chunk keeps the original ``source`` id; later windows use ``_1``, ``_2``, …
    Prefix gold matching in ``eval_quality_metrics`` treats all as relevant for one abstract.
    """
    step = max(1, chunk_words - overlap_words)
    written = 0
    with open(input_jsonl, encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            source = str(obj.get("source") or "")
            words = text.split()

            if len(words) <= chunk_words:
                rec = {
                    "text": text,
                    "source": source,
                    "chunk_id": source or f"chunk_{written}",
                    "char_offset": 0,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            else:
                i, idx, off = 0, 0, 0
                while i < len(words):
                    window = words[i : i + chunk_words]
                    if len(window) < 20:
                        break
                    sub_source = source if idx == 0 else f"{source}_{idx}"
                    body = " ".join(window)
                    fout.write(
                        json.dumps(
                            {
                                "text": body,
                                "source": sub_source,
                                "chunk_id": sub_source,
                                "char_offset": off,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    off += len(body)
                    idx += 1
                    written += 1
                    i += step
    print(f"build_fixed_chunks: {written} chunks -> {output_jsonl!r}", flush=True)
    return written


def _write_pubmedqa_gold_chunk_lines(
    f: Any,
    item: Dict[str, Any],
    *,
    chunk_type: str,
) -> int:
    """Write abstract (+ optional long_answer) lines; return line count."""
    pubid = str(item.get("id", "")).strip()
    context = str(item.get("context") or "").strip()
    long_answer = str(item.get("long_answer") or "").strip()
    if not pubid or not context:
        return 0
    n_lines = 0
    f.write(
        json.dumps(
            {
                "source": f"pubmedqa_{pubid}",
                "text": context,
                "pubid": pubid,
                "chunk_type": chunk_type,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    n_lines += 1
    if long_answer and len(long_answer) > 50:
        f.write(
            json.dumps(
                {
                    "source": f"pubmedqa_{pubid}_long",
                    "text": long_answer,
                    "pubid": pubid,
                    "chunk_type": f"{chunk_type}_long_answer",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        n_lines += 1
    return n_lines


def build_pubmedqa_gold_index(out_dir: str, max_items: int = 0) -> str:
    """
    Build ``chunks.jsonl`` with ``pubmedqa_<pubid>`` sources for retrieval eval.

    Includes **eval holdout** abstracts (for recall@k targets) plus **corpus** abstracts
  (distractors). PubMedQA **generation** still uses the primary external index only (C4).
    """
    os.makedirs(out_dir, exist_ok=True)
    state = _ensure_pubmedqa_split_loaded()
    corpus_items = list(state["corpus_items"])
    eval_items = list(state["eval_items"])
    if max_items > 0 and len(corpus_items) > max_items:
        corpus_items = random.Random(42).sample(corpus_items, max_items)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    n_lines = 0
    with open(chunks_path, "w", encoding="utf-8") as f:
        for item in corpus_items:
            n_lines += _write_pubmedqa_gold_chunk_lines(f, item, chunk_type="corpus_holdout")
        for item in eval_items:
            n_lines += _write_pubmedqa_gold_chunk_lines(f, item, chunk_type="eval_holdout")
    manifest_path = _write_pubmedqa_gold_manifest(out_dir, state)
    missing = _gold_index_missing_eval_pubids(out_dir, _pubmedqa_eval_pubids())
    if missing:
        raise RuntimeError(
            f"Gold index at {out_dir!r} is missing {len(missing)} eval-holdout PMIDs "
            f"(e.g. {missing[:5]}). recall@k cannot be computed."
        )
    print(
        f"build_pubmedqa_gold_index: {n_lines} chunks "
        f"({len(corpus_items)} corpus + {len(eval_items)} eval holdout PMIDs) -> {chunks_path!r}\n"
        f"  manifest: {manifest_path!r}\n"
        f"  PubMedQA task RAG uses primary external index; gold index is for recall@k eval only.",
        flush=True,
    )
    return os.path.abspath(out_dir)


def _build_faiss_inprocess_from_jsonl(
    chunks_path: str, out_dir: str, embed_model: str, *, batch_size: int = 32
) -> None:
    """Write ``index.faiss`` for an existing ``chunks.jsonl`` (Kaggle paste-only fallback)."""
    import faiss  # type: ignore
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts: List[str] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj.get("text", "")).strip())
    if not any(texts):
        raise ValueError(f"No non-empty text in {chunks_path!r}")
    st = SentenceTransformer(embed_model)
    dim = st.get_sentence_embedding_dimension()
    batches: List[Any] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            emb = st.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            emb = st.encode(batch, convert_to_numpy=True)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        faiss.normalize_L2(emb)
        batches.append(emb)
    mat = np.vstack(batches)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    if index.ntotal != len(texts):
        raise RuntimeError(f"FAISS ntotal={index.ntotal} vs chunks={len(texts)}")
    idx_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    print(
        f"Gold FAISS (in-process): {idx_path} ntotal={index.ntotal} dim={dim}",
        flush=True,
    )


def _build_faiss_for_gold_chunks(out_dir: str, embed_model: str) -> None:
    """Embed gold ``chunks.jsonl`` and write ``index.faiss`` (CLI or in-process fallback)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_rag_index.py")
    chunks = os.path.join(out_dir, "chunks.jsonl")
    if not os.path.isfile(chunks):
        raise FileNotFoundError(f"Gold chunks missing: {chunks!r}")
    if os.path.isfile(script):
        cmd = [
            sys.executable,
            script,
            "--input",
            chunks,
            "--out_dir",
            out_dir,
            "--embed_model",
            embed_model,
        ]
        print(f"Gold FAISS build: {' '.join(cmd)}", flush=True)
        try:
            subprocess.check_call(cmd)
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            print(
                f"Gold FAISS subprocess failed ({exc}); retrying in-process embed.",
                flush=True,
            )
    else:
        print(
            f"Gold FAISS: {script!r} not on disk; using in-process embed (paste-only Kaggle OK).",
            flush=True,
        )
    _build_faiss_inprocess_from_jsonl(chunks, out_dir, embed_model)


def build_pubmedqa_gold_rag_index(
    out_dir: str,
    max_items: int = 0,
    *,
    build_faiss: bool = True,
    embed_model: str = "",
) -> str:
    """Build ``chunks.jsonl`` + optional ``index.faiss`` with ``pubmedqa_<pubid>`` sources."""
    d = build_pubmedqa_gold_index(out_dir, max_items=max_items)
    if build_faiss:
        em = (
            embed_model.strip()
            or os.environ.get("RAG_EMBED_MODEL", "").strip()
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        _build_faiss_for_gold_chunks(d, em)
    return d


def _apply_pubmedqa_gold_rag_from_args(ns: Any) -> str:
    """Set ``GP_RAG_GOLD_INDEX_DIR`` for PubMedQA-only retrieval when gold index is ready."""
    gold = ""
    if getattr(ns, "rag_gold_index_dir", "").strip():
        gold = os.path.abspath(ns.rag_gold_index_dir.strip())
    elif getattr(ns, "auto_gold_pubmedqa", False):
        cand = _resolve_gold_rag_dir()
        if cand:
            gold = cand
        else:
            build_at = _default_gold_rag_build_dir()
            print(
                f"\n--auto_gold_pubmedqa: no ready gold index found.\n"
                f"  Attach Kaggle dataset salmashopna/rag-index-gold, or run:\n"
                f"  python eval_benchmarks.py --build_gold_index {build_at!r}\n",
                flush=True,
            )
    if gold:
        if not _gold_rag_index_ready(gold):
            print(
                f"WARNING: gold RAG dir {gold!r} needs chunks.jsonl + index.faiss with pubmedqa_* rows.",
                flush=True,
            )
            os.environ.pop("GP_RAG_GOLD_INDEX_DIR", None)
            return ""
        os.environ["GP_RAG_GOLD_INDEX_DIR"] = gold
        gcp = os.path.join(gold, "chunks.jsonl")
        n_total, _is_seed, pubmed_hits = _rag_chunks_metadata(gcp)
        print(
            f"RAG gold (PubMedQA RAG + recall@k): {gold!r} — {pubmed_hits} pubmedqa_* / {n_total} chunks. "
            "MedQA/MMLU use the primary index.",
            flush=True,
        )
        return gold
    os.environ.pop("GP_RAG_GOLD_INDEX_DIR", None)
    return ""


def _rag_row_diagnostic_fields(
    benchmark: str,
    item: Dict[str, Any],
    use_rag: bool,
    ranked_sources: List[str],
    rag_hits: List[Dict[str, Any]],
    rag_diagnostic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Step 07 per-sample retrieval diagnostics."""
    if not use_rag:
        return {}
    diag = dict(rag_diagnostic or {})
    rel = _rag_relevant_source_ids(item, benchmark)
    ranked_sources = _ranked_sources_for_metrics(ranked_sources, rag_hits, diag)
    chunk_ids = _chunk_ids_from_rag(rag_hits, ranked_sources, diag)
    if chunk_ids and not diag.get("retrieved_chunk_ids"):
        diag["retrieved_chunk_ids"] = chunk_ids
    gold_rank: Optional[int] = None
    gold_found = False
    if rel and ranked_sources:
        from eval_quality_metrics import doc_matches_relevant

        for rank, src in enumerate(ranked_sources, start=1):
            for g in rel:
                if doc_matches_relevant(str(src), g):
                    gold_rank = rank
                    gold_found = True
                    break
            if gold_found:
                break
    return {
        "retrieval_method": diag.get("retrieval_method", ""),
        "retrieved_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
        "similarity_scores": json.dumps(diag.get("similarity_scores") or [], ensure_ascii=False),
        "reranker_scores": json.dumps(diag.get("reranker_scores") or [], ensure_ascii=False),
        "gold_chunk_rank": gold_rank if gold_rank is not None else "",
        "gold_chunk_found": gold_found,
        "context_char_length": diag.get("context_char_length", ""),
        "retrieval_latency_ms": diag.get("retrieval_latency_ms", ""),
    }


def _parse_rag_relevant_sources(raw: Any) -> Set[str]:
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw if str(x)}
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return {str(x) for x in parsed if str(x)}
            except json.JSONDecodeError:
                pass
        if s:
            return {s}
    return set()


def _print_retrieval_health_validation(prediction_rows: List[Dict[str, Any]]) -> None:
    """Step 08: warn on binary recall@k collapse for PubMedQA RAG rows."""
    from eval_quality_metrics import compute_retrieval_metrics

    try:
        from rag_retrieval import retrieval_health_summary
    except ImportError:
        return
    per_query: List[Dict[str, float]] = []
    for r in prediction_rows:
        if not r.get("rag_flag") or not r.get("retrieval_evaluable"):
            continue
        diag: Dict[str, Any] = {}
        for key in ("retrieved_chunk_ids",):
            raw = r.get(key)
            if isinstance(raw, str) and raw.strip().startswith("["):
                try:
                    diag[key] = json.loads(raw)
                except json.JSONDecodeError:
                    pass
        ranked = _ranked_sources_for_metrics(
            list(r.get("rag_ranked_sources") or []),
            r.get("rag_hits") if isinstance(r.get("rag_hits"), list) else [],
            diag,
        )
        rel = _parse_rag_relevant_sources(r.get("rag_relevant_sources"))
        if not rel:
            continue
        per_query.append(compute_retrieval_metrics(ranked, rel, ks=_rag_recall_ks()))
    if not per_query:
        return
    health = retrieval_health_summary(per_query, ks=_rag_recall_ks())
    print("\n=== Retrieval health (Step 08) ===", flush=True)
    for k in _rag_recall_ks():
        mk = f"recall_at_{k}_mean"
        if mk in health:
            print(f"  {mk}: {health[mk]}", flush=True)
    if health.get("mrr_mean") is not None:
        print(f"  mrr_mean: {health['mrr_mean']}", flush=True)
    ks = _rag_recall_ks()
    if _recall_means_all_zero(health, ks) and _external_corpus_retrieval_metrics_na():
        print(f"  {_RETRIEVAL_NA_LOG_MSG}", flush=True)
        health["retrieval_metrics_na"] = True
        health["retrieval_na_reason"] = "external_non_leaking_corpus_no_pubmedqa_gold"
    elif health.get("binary_collapse_detected"):
        if _recall_means_any_positive(health, ks):
            print(
                "  WARNING: binary collapse (recall@1 ≈ recall@3 ≈ recall@5 > 0, flat spread). "
                "Rebuild corpus with build_fixed_chunks + build_rag_index; ensure hybrid retrieval.",
                flush=True,
            )
        else:
            print(
                "  NOTE: recall@k is 0 for all evaluable PubMedQA rows — no gold hit in retrieved ranks. "
                "RAG retrieval still ran (see rag_source=faiss). Use a gold-aligned index for recall@k.",
                flush=True,
            )
    elif health.get("healthy"):
        print("  PASS: differentiated recall@k (monotonic spread).", flush=True)
    return health


def _print_rag_post_run_diagnostic(prediction_rows: List[Dict[str, Any]]) -> None:
    """Warn when every RAG row used lexical fallback (FAISS/hybrid not active)."""
    rag_rows = [r for r in prediction_rows if r.get("rag_flag")]
    if not rag_rows:
        return
    from collections import Counter

    src_counts = Counter(str(r.get("rag_source") or "") for r in rag_rows)
    dense_like = sum(
        src_counts.get(k, 0) for k in ("faiss", "hybrid", "dense", "hybrid+rerank", "dense+rerank")
    )
    if src_counts.get("lexical", 0) == len(rag_rows):
        print(
            "\nRAG DIAGNOSTIC: ALL RAG rows used lexical fallback (rag_source='lexical').\n"
            "  FAISS/hybrid was NOT used (check faiss_import_error in rag_index_verify).\n"
            "  Fix: pip install faiss-cpu, restart kernel, verify: import faiss\n"
            "  Then re-run !python eval_benchmarks.py (do not use --allow_lexical_rag).\n",
            flush=True,
        )
    elif dense_like > 0:
        pct = dense_like / len(rag_rows) * 100.0
        print(f"\nRAG DIAGNOSTIC: {pct:.1f}% of RAG rows used dense/hybrid retrieval.", flush=True)


def _evaluate_mcq_for_model(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_key: str,
    model_ids: Optional[Dict[str, str]] = None,
    active_configs: Optional[List[Tuple[str, str, bool]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """NoRAG + RAG for one model while that model stays on GPU."""
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    mids = model_ids or _model_ids_snapshot()
    cfgs = active_configs if active_configs is not None else CONFIGS
    for cfg_name, mk, use_rag in cfgs:
        if mk != model_key:
            continue
        preds: List[str] = []
        golds: List[str] = []
        f1s: List[float] = []
        row_start = len(rows)
        for item in _item_iterator(items, desc=f"{benchmark}/{cfg_name}"):
            choices = [c for c in item["choices"] if c is not None]
            n = len(choices)
            gold = _normalize_mcq_gold(str(item["answer"]), n)
            gold_text = _mcq_choice_text(choices, gold)
            rctx, rhits, rranked, rsrc, rdiag = "", [], [], "", {}
            effective_rag = False
            if use_rag:
                rctx, rhits, rranked, rsrc, rdiag = _prefetch_rag_for_query(item["question"])
                rhits = _filter_rag_hits_by_score(rhits)
                effective_rag = _should_use_rag_context(rhits, rctx)
            prompt = _build_mcq_prompt(
                item["question"],
                choices,
                context=rctx if effective_rag else "",
                use_rag=effective_rag,
            )
            prev_mcq_gen = os.environ.get("GEN_MAX_NEW_TOKENS")
            if effective_rag or (use_rag and model_key == "llm"):
                os.environ["GEN_MAX_NEW_TOKENS"] = str(_mcq_gen_max_new_tokens(model_key, effective_rag))
            try:
                out = run_fn(model_key, prompt, False)
            finally:
                if prev_mcq_gen is None:
                    os.environ.pop("GEN_MAX_NEW_TOKENS", None)
                else:
                    os.environ["GEN_MAX_NEW_TOKENS"] = prev_mcq_gen
            raw, _rctx2, _rsrc2, _rh2, _rr2, _rd2 = _run_out(out)
            telem = _inference_telemetry(out)
            if use_rag:
                if not rctx:
                    rctx, rhits, rranked, rsrc = _rctx2, _rh2, _rr2, _rsrc2
                if _rd2 and not rdiag.get("retrieved_chunk_ids"):
                    rdiag = {**rdiag, **_rd2}
                if not rranked and _rr2:
                    rranked = _rr2
                if not rhits and _rh2:
                    rhits = _rh2
            pred = _extract_mcq_letter(raw, n)
            preds.append(pred)
            golds.append(gold)
            pred_text = _mcq_choice_text(choices, pred)
            f1_val, f1_label = _mcq_token_f1(pred, gold, choices)
            f1s.append(f1_val)
            row: Dict[str, Any] = {
                    "benchmark": benchmark,
                    "question_id": str(item.get("id", "")),
                    "question": item["question"],
                    "reference_answer": gold,
                    "reference_text": gold_text,
                    "prediction_text": pred_text,
                    "model_name": cfg_name,
                    "model_key": model_key,
                    "rag_flag": use_rag,
                    "rag_context_used": effective_rag,
                    "rag_context_rejected": bool(use_rag and not effective_rag),
                    "model_answer": pred,
                    "raw_response": raw,
                    "retrieved_context": rctx if use_rag else "",
                    "rag_source": rsrc if use_rag else "",
                "rag_hits": rhits if use_rag else [],
                "rag_ranked_sources": rranked if use_rag else [],
                    "parsed_prediction": pred,
                    "mcq_correct": pred == gold,
                "label_correct": pred == gold,
                    "choices_json": json.dumps(choices, ensure_ascii=False),
                "token_f1": f1_val,
                "token_f1_label": f1_label,
                }
            row.update(_row_model_meta(model_key, mids))
            row.update(telem)
            row.update(
                _retrieval_metrics_fields(benchmark, item, use_rag, rranked, rhits, rdiag)
            )
            row.update(_rag_row_diagnostic_fields(benchmark, item, use_rag, rranked, rhits, rdiag))
            rows.append(row)
        if benchmark == "mmlu_med":
            cluster_ids = [str(item.get("subject") or str(item.get("id", "")).split("_")[0]) for item in items]
            mean, lo, hi = _accuracy_cluster_bootstrap(preds, golds, cluster_ids)
        else:
            mean, lo, hi = _accuracy(preds, golds)
        fm, fl, fh = _bootstrap_ci(f1s) if f1s else (float("nan"),) * 3
        acc_bits = [1.0 if p == g else 0.0 for p, g in zip(preds, golds)]
        _, acc_std = _binary_mean_std(acc_bits)
        _, f1_std = _binary_mean_std(f1s) if f1s else (float("nan"), float("nan"))
        results[cfg_name] = {
            "metric": "accuracy",
            "mean": mean,
            "std": acc_std,
            "ci_lower": lo,
            "ci_upper": hi,
            "accuracy_ci_method": "cluster_bootstrap_by_subject" if benchmark == "mmlu_med" else "item_bootstrap",
            "f1_mean": fm,
            "f1_std": f1_std,
            "f1_ci_lower": fl,
            "f1_ci_upper": fh,
            "n": len(items),
        }
        if use_rag:
            results[cfg_name].update(_config_retrieval_summary(rows[row_start:]))
    _validate_token_f1_distribution(rows, benchmark)
    return results, rows


def _evaluate_mcq(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_ids: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for mk in ("slm", "llm"):
        r, rw = _evaluate_mcq_for_model(items, run_fn, benchmark, mk, model_ids=model_ids)
        results.update(r)
        rows.extend(rw)
    return results, rows


def _evaluate_free_text_for_model(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_key: str,
    model_ids: Optional[Dict[str, str]] = None,
    active_configs: Optional[List[Tuple[str, str, bool]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    use_pubmed_prompt = benchmark == "pubmedqa"
    mids = model_ids or _model_ids_snapshot()
    cfgs = active_configs if active_configs is not None else CONFIGS

    for cfg_name, mk, use_rag in cfgs:
        if mk != model_key:
            continue
        preds: List[str] = []
        refs: List[str] = []
        f1s: List[float] = []
        label_hits: List[float] = []
        row_start = len(rows)

        for item in _item_iterator(items, desc=f"{benchmark}/{cfg_name}"):
            q = item["question"]
            ref_label, ref_text = _pubmedqa_metric_reference(item)
            abstract = str(item.get("context") or "")
            rctx, rhits, rranked, rsrc, rdiag = "", [], [], "", {}
            effective_rag = False
            if use_rag and use_pubmed_prompt:
                rctx, rhits, rranked, rsrc, rdiag = _prefetch_rag_for_pubmedqa(item)
                rhits = _filter_rag_hits_by_score(rhits)
                effective_rag = _should_use_rag_context(rhits, rctx)
            if use_pubmed_prompt:
                prompt = _pubmedqa_prompt_with_optional_rag(
                    q,
                    abstract,
                    rctx if effective_rag else "",
                    model_key=model_key,
                )
            else:
                prompt = f"{q}\n\nAnswer briefly: yes, no, or maybe only."

            if use_pubmed_prompt:
                prev_gen = os.environ.get("GEN_MAX_NEW_TOKENS")
                os.environ["GEN_MAX_NEW_TOKENS"] = str(
                    _pubmedqa_max_new_tokens(model_key, effective_rag)
                )
                try:
                    out = run_fn(model_key, prompt, False)
                finally:
                    if prev_gen is None:
                        os.environ.pop("GEN_MAX_NEW_TOKENS", None)
                    else:
                        os.environ["GEN_MAX_NEW_TOKENS"] = prev_gen
            else:
                out = run_fn(model_key, prompt, use_rag)
            raw, rctx_run, rsrc_run, rhits_run, rranked_run, rdiag_run = _run_out(out)
            telem = _inference_telemetry(out)
            if use_rag:
                if not rctx:
                    rctx, rhits, rranked, rsrc = rctx_run, rhits_run, rranked_run, rsrc_run
                if rdiag_run and not rdiag.get("retrieved_chunk_ids"):
                    rdiag = {**rdiag, **rdiag_run}
                if not rranked and rranked_run:
                    rranked = rranked_run
                if not rhits and rhits_run:
                    rhits = rhits_run
                if not use_pubmed_prompt:
                    effective_rag = True
            preds.append(raw)
            refs.append(ref_text)

            parsed = _parse_pubmed_model_answer(raw) if use_pubmed_prompt else ""
            if use_pubmed_prompt:
                row_parse_status = _pubmedqa_parse_status(raw, parsed)
            f1_main, f1_lbl = (
                _pubmedqa_token_f1(raw, parsed, ref_label, ref_text)
                if use_pubmed_prompt
                else (_token_f1(raw, ref_text), 0.0)
            )
            f1s.append(f1_main)
            if use_pubmed_prompt:
                if ref_label in ("yes", "no", "maybe") and parsed:
                    lc = parsed == ref_label
                    label_hits.append(1.0 if lc else 0.0)
                else:
                    lc = False
                    label_hits.append(0.0)
            else:
                lc = False
                label_hits.append(float("nan"))

            ctx_full = str(item.get("context") or "")
            row = {
                    "benchmark": benchmark,
                    "question_id": str(item.get("id", "")),
                    "question": q,
                    "reference_answer": ref_label,
                    "reference_text": ref_text,
                    "context": ctx_full,
                    "context_chars": len(str(item.get("context") or "")),
                    "model_name": cfg_name,
                    "model_key": model_key,
                    "rag_flag": use_rag,
                    "model_answer": raw,
                    "raw_response": raw,
                    "retrieved_context": rctx if use_rag else "",
                    "rag_source": rsrc if use_rag else "",
                "rag_hits": rhits if use_rag else [],
                "rag_ranked_sources": rranked if use_rag else [],
                    "parsed_prediction": parsed,
                "label_correct": lc,
                "mcq_correct": lc if use_pubmed_prompt else "",
                    "choices_json": "",
                    "token_f1": f1s[-1],
                "token_f1_label": f1_lbl if use_pubmed_prompt else "",
                "rag_context_used": effective_rag if use_pubmed_prompt else (use_rag and bool(rctx)),
                "rag_context_rejected": bool(use_rag and use_pubmed_prompt and not effective_rag),
                }
            row.update(_row_model_meta(model_key, mids))
            row.update(telem)
            row.update(
                _retrieval_metrics_fields(benchmark, item, use_rag, rranked, rhits, rdiag)
            )
            row.update(_rag_row_diagnostic_fields(benchmark, item, use_rag, rranked, rhits, rdiag))
            if use_rag and use_pubmed_prompt and _pubmedqa_retrieval_index_dir():
                row["rag_gold_index_dir"] = _pubmedqa_retrieval_index_dir()
            if use_pubmed_prompt:
                row["pubmedqa_parse_status"] = row_parse_status
                row["pubmedqa_eval_split"] = _pubmedqa_eval_split_descriptor()
                row["pubmedqa_retrieval_query_mode"] = (
                    "question_plus_abstract"
                    if _pubmedqa_retrieval_query_includes_abstract()
                    else "question_only"
                )
            if use_rag and effective_rag and rctx:
                row["evidence_token_overlap"] = round(
                    _evidence_token_overlap(rctx, raw), 4
                )
            rows.append(row)

        fm, fl, fh = _bootstrap_ci(f1s)
        out: Dict[str, Any] = {
            "metric": "free_text",
            "f1_mean": fm,
            "f1_ci_lower": fl,
            "f1_ci_upper": fh,
            "n": len(items),
        }
        if use_pubmed_prompt:
            out["pubmedqa_metrics_mode"] = (
                "label_accuracy_only"
                if not _pubmedqa_generative_metrics_enabled()
                else "label_accuracy_plus_generative"
            )
            out["pubmedqa_eval_split"] = _pubmedqa_eval_split_descriptor()
            if not _pubmedqa_generative_metrics_enabled():
                out["generative_metrics_skipped"] = (
                    "PubMedQA uses one-word yes/no/maybe outputs; "
                    "BLEU/ROUGE/METEOR/BERTScore vs long_answer are not reported (audit C5)."
                )
            else:
                rm, rl, rh = _rouge_l(preds, refs)
                out["rougeL_mean"] = rm
                out["rougeL_ci_lower"] = rl
                out["rougeL_ci_upper"] = rh
                _merge_metric_ci(out, "bleu", _bleu(preds, refs))
                _merge_metric_ci(out, "meteor", _meteor(preds, refs))
                bsc = _bertscore_f1(preds, refs)
                _merge_metric_ci(out, "bertscore_f1", bsc[:3])
                out["bertscore_model_type"] = bsc[3]
            clean_hits = [x for x in label_hits if x == x]
            if clean_hits:
                lm, ll, lh = _bootstrap_ci(clean_hits)
                _, lab_std = _binary_mean_std(clean_hits)
                out["pubmedqa_label_accuracy_mean"] = lm
                out["pubmedqa_label_accuracy_std"] = lab_std
                out["pubmedqa_label_accuracy_ci_lower"] = ll
                out["pubmedqa_label_accuracy_ci_upper"] = lh
                out["mcq_accuracy_mean"] = lm
                out["mcq_accuracy_std"] = lab_std
        else:
            rm, rl, rh = _rouge_l(preds, refs)
            out["rougeL_mean"] = rm
            out["rougeL_ci_lower"] = rl
            out["rougeL_ci_upper"] = rh
            _merge_metric_ci(out, "bleu", _bleu(preds, refs))
            _merge_metric_ci(out, "meteor", _meteor(preds, refs))
            bsc = _bertscore_f1(preds, refs)
            _merge_metric_ci(out, "bertscore_f1", bsc[:3])
            out["bertscore_model_type"] = bsc[3]
        if use_rag:
            out.update(_config_retrieval_summary(rows[row_start:]))
        results[cfg_name] = out
    _validate_token_f1_distribution(rows, benchmark)
    return results, rows


def _evaluate_free_text(
    items: List[Dict[str, Any]],
    run_fn: Callable[[str, str, bool], Any],
    benchmark: str,
    model_ids: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for mk in ("slm", "llm"):
        r, rw = _evaluate_free_text_for_model(items, run_fn, benchmark, mk, model_ids=model_ids)
        results.update(r)
        rows.extend(rw)
    return results, rows


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
        items = _load_mmlu_med()
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
        sec_rows = _load_mmlu_med()
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
    """Write ``{prefix}_predictions.json`` and ``{prefix}_predictions.csv``. PubMedQA (and any row
    with dataset ``context``) includes that text in JSON/CSV for downstream RAG corpus export."""
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
    fieldnames: List[str] = []
    seen: set[str] = set()
    for col in _PREDICTION_CSV_COLUMNS:
        seen.add(col)
        fieldnames.append(col)
    for row in all_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    if not all_rows:
        with open(cpath, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
        print(f"Wrote {jpath} (0 rows) and {cpath} (header only)", flush=True)
        return
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
    hf_router: bool = False,
    hf_router_base_url: str = "",
    hf_router_model_slm: str = "",
    hf_router_model_llm: str = "",
) -> Dict[str, Any]:
    # Import HF datasets before prepending /kaggle/working (avoids shadowing pandas/datasets).
    _import_load_dataset()

    for d in (_work_dir(),):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here and here not in sys.path:
            sys.path.insert(0, here)
    except NameError:
        pass

    if mock and "pubmedqa" in benchmarks:
        raise RuntimeError(
            "PubMedQA must use real model outputs (full abstract + yes/no/maybe). "
            "Re-run without --mock, or drop pubmedqa from this run."
        )

    if mock and hf_router:
        raise RuntimeError("--mock and --hf_router cannot be used together.")

    inference_backend = "local_gpu"
    hf_router_meta: Optional[Dict[str, str]] = None

    if mock:
        inference_backend = "mock"
        print("MOCK: no GPU models (MCQ-only fake answers).", flush=True)

        def run_fn(model_key: str, question: str, use_rag: bool) -> str:
            return "A" if "OPTIONS" in question.upper() else "yes"

    elif hf_router:
        inference_backend = "hf_router"
        _setup_runner_path()
        _drop_runner_caches()
        _ensure_hf_token()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        import real_model_runner as _rmr

        base = (
            (hf_router_base_url or os.environ.get("HF_ROUTER_BASE_URL") or "").strip()
            or "https://router.huggingface.co/v1"
        ).rstrip("/")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token or not str(token).strip():
            raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) required for --hf_router.")

        mid_slm, mid_llm = _resolve_hf_router_models(hf_router_model_slm, hf_router_model_llm)
        client = OpenAI(api_key=str(token).strip(), base_url=base)
        hf_router_meta = {"base_url": base, "model_slm": mid_slm, "model_llm": mid_llm}
        print(
            f"HF Inference Router (OpenAI API): base_url={base!r} SLM={mid_slm!r} LLM={mid_llm!r} "
            "(no local GPU; RAG via build_rag_context).",
            flush=True,
        )

        def run_fn(model_key: str, question: str, use_rag: bool) -> Dict[str, Any]:
            # Eval prefetches RAG into the prompt and always passes use_rag=False (audit M5).
            if use_rag:
                print(
                    "GP_BENCH: hf_router ignores use_rag=True; embed RAG in the prompt (eval prefetch).",
                    flush=True,
                )
                use_rag = False
            rag_block, _ev_snip, src = _rmr.build_rag_context(question, use_rag)
            pubmed_style = "one word only" in (question or "").lower()
            if use_rag and rag_block:
                user_content = f"Medical Context: {rag_block}\n\nQuery: {question}\n\nClinical Answer:"
            else:
                user_content = f"Medical Query: {question}\n\nClinical Answer:"
            model_id = mid_slm if model_key == "slm" else mid_llm
            comp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=_hf_router_max_tokens(pubmed_one_word=pubmed_style),
                temperature=0,
            )
            text = (comp.choices[0].message.content or "").strip()
            return {
                "response": text,
                "retrieved_context": _rmr.LAST_RAG_EVIDENCE if use_rag else "",
                "rag_source": src if use_rag else "none",
                "rag_hits": list(_rmr.LAST_RAG_HITS) if use_rag else [],
                "rag_ranked_sources": list(_rmr.LAST_RAG_RANKED_SOURCES) if use_rag else [],
            }

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
            "Models load one-at-a-time: each SLM/LLM stays loaded across all benchmarks "
            "(lower peak VRAM; avoids reload per dataset).",
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

    _set_bootstrap_rng(subset_seed)

    agg: Dict[str, Any] = {
        "max_items_per_benchmark": max_items,
        "subset_seed": subset_seed,
        "bootstrap_ci_seed": (int(subset_seed) + 17) if subset_seed is not None else None,
        "use_4bit": use_4bit,
        "mock": mock,
        "inference_backend": inference_backend,
        "benchmarks": {},
        "quantization_disclosure": {
            "use_4bit": use_4bit,
            "default_on_cuda": True,
            "paper_recommendation": (
                "Report --no_4bit fp16/bf16 primary results or both with accuracy delta."
            ),
        },
        "dataset_provenance": _dataset_provenance_snapshot(benchmarks),
        "multiple_comparisons": _multiple_comparisons_disclosure(benchmarks),
        "gp_bundle_version": GP_BUNDLE_VERSION,
    }
    if hf_router_meta is not None:
        agg["hf_router"] = hf_router_meta
        agg["hf_router"]["rag_mode"] = "eval_prefetch_in_prompt_only"
    if rag_config:
        agg["rag_config"] = rag_config

    model_ids = _model_ids_snapshot(hf_router_meta)
    agg["model_ids"] = model_ids
    print(
        f"Models: SLM={model_ids['slm_model_id']!r} LLM={model_ids['llm_model_id']!r} "
        f"use_4bit={use_4bit} subset_seed={subset_seed} mock={mock}",
        flush=True,
    )

    if not mock and any(c[2] for c in CONFIGS):
        try:
            _setup_runner_path()
            _drop_runner_caches()
            import real_model_runner as rmr

            rmr.ensure_rag_index_env()
            rmr.print_startup_diagnostics()
            faiss_status = rmr.verify_rag_index(require_faiss=False)
            agg["rag_index_verify"] = faiss_status
            if not faiss_status.get("faiss_deps_ok"):
                err = faiss_status.get("faiss_import_error") or "unknown"
                msg = (
                    "faiss-cpu not available — RAG will use lexical fallback only. "
                    f"faiss_import_error={err!r}. "
                    "Cell 1b: pip install faiss-cpu, restart, verify import faiss. "
                    "Or use !python after restart (not Run Cell on this .py file)."
                )
                if _kaggle_strict_checks():
                    raise RuntimeError(msg)
                print(f"RAG WARNING: {msg}", flush=True)
            elif faiss_status.get("faiss_load_ok"):
                if not _prewarm_rag_faiss():
                    print(
                        "RAG WARNING: FAISS prewarm failed — check transformers>=4.43, faiss-cpu, "
                        "sentence-transformers, and RAG_INDEX_DIR before benchmarks run.",
                        flush=True,
                    )
        except Exception as ex:
            print(f"RAG startup check skipped: {ex}", flush=True)

    rag_dir = ""
    if rag_config:
        rag_dir = str(rag_config.get("RAG_INDEX_DIR") or "").strip()
    if not rag_dir:
        rag_dir = os.environ.get("RAG_INDEX_DIR", "").strip()
    auto_gold = _auto_enable_gold_pubmedqa_index_if_needed(benchmarks)
    gold_dir = os.environ.get("GP_RAG_GOLD_INDEX_DIR", "").strip()
    if "pubmedqa" in benchmarks:
        split_st = _ensure_pubmedqa_split_loaded()
        agg["pubmedqa_split"] = {
            "eval_split_descriptor": split_st.get("eval_split_descriptor"),
            "n_all_labeled": split_st.get("n_all"),
            "n_eval_holdout": split_st.get("n_eval"),
            "n_corpus_for_gold_index": split_st.get("n_corpus"),
            "holdout_seed": _pubmedqa_holdout_seed(),
            "holdout_frac": _pubmedqa_holdout_frac(),
            "not_official_pubmedqa_test_split": True,
        }
    agg["rag_experiment"] = _build_rag_experiment_manifest(benchmarks)
    agg["rag_experiment"]["auto_gold_pubmedqa"] = auto_gold
    agg["energy_measurement_disclosure"] = _energy_measurement_disclosure()
    agg["retrieval_validation"] = _validate_retrieval_labels_before_run(
        benchmarks, rag_dir, gold_dir
    )

    prediction_rows: List[Dict[str, Any]] = []
    mcq_benches = {"medqa", "medmcqa", "mmlu_med", "custom_mcq"}
    bench_items: Dict[str, List[Dict[str, Any]]] = {}

    for name in benchmarks:
        dp = data_paths.get(name, "")
        print(f"\n=== {name} === (loading)", flush=True)
        items = _load_items(name, dp, max_items, subset_seed)
        bench_items[name] = items
        print(f"  n={len(items)}", flush=True)
        entry: Dict[str, Any] = {"n_items": len(items)}
        if name == "mmlu_med":
            entry["mmlu_split"] = _mmlu_med_split()
            entry["mmlu_pretraining_contamination_risk"] = (
                "high on test split; default dev reduces but does not eliminate risk"
            )
        if name == "pubmedqa":
            st = _ensure_pubmedqa_split_loaded()
            entry["pubmedqa_eval_split"] = st.get("eval_split_descriptor")
        if not items:
            entry["error"] = "no_items"
        agg["benchmarks"][name] = entry

    for model_key in ("slm", "llm"):
        print(f"\n--- Model block: {model_key.upper()} (all benchmarks) ---", flush=True)
        for name in benchmarks:
            items = bench_items.get(name) or []
            if not items:
                continue
            print(f"\n=== {name} === [{model_key.upper()}]", flush=True)
            try:
                if name in mcq_benches:
                    res, rows = _evaluate_mcq_for_model(
                        items, run_fn, name, model_key, model_ids=model_ids
                    )
                else:
                    res, rows = _evaluate_free_text_for_model(
                        items, run_fn, name, model_key, model_ids=model_ids
                    )
                prediction_rows.extend(rows)
                prev = agg["benchmarks"].get(name) or {}
                if prev.get("results"):
                    prev["results"].update(res)
                    merged = prev["results"]
                else:
                    merged = dict(res)
                entry = {k: v for k, v in prev.items() if k != "results"}
                entry["n_items"] = len(items)
                entry["results"] = merged
                agg["benchmarks"][name] = entry
                rag_delta = _check_rag_accuracy_delta(merged, name)
                bench_rows = [r for r in prediction_rows if str(r.get("benchmark")) == name]
                rag_delta.update(_paired_rag_significance(bench_rows, name))
                agg["benchmarks"][name]["rag_accuracy_delta"] = rag_delta
            except Exception as ex:
                agg["benchmarks"][name] = {"error": str(ex)}
                print(f"  ERROR: {ex}", flush=True)

    if save_predictions_prefix.strip():
        pred_meta = {
            "benchmarks": benchmarks,
            "max_items": max_items,
            "subset_seed": subset_seed,
            "mock": mock,
            "use_4bit": use_4bit,
            "inference_backend": inference_backend,
            "configs": [c[0] for c in CONFIGS],
            "model_ids": model_ids,
        }
        if hf_router_meta is not None:
            pred_meta["hf_router"] = hf_router_meta
        _write_prediction_artifacts(
            save_predictions_prefix.strip(),
            prediction_rows,
            pred_meta,
        )
        _print_rag_post_run_diagnostic(prediction_rows)

    if prediction_rows:
        health = _print_retrieval_health_validation(prediction_rows)
        if health:
            agg["retrieval_health"] = health
        agg["rag_context_rejection"] = _rag_context_rejection_summary(prediction_rows)
        agg["rag_context_rejection_by_benchmark"] = _rag_context_rejection_by_benchmark(
            prediction_rows
        )
        agg["paired_rag_significance"] = _paired_rag_significance_all(prediction_rows)
        agg["pubmedqa_parse_diagnostics"] = _pubmedqa_parse_failure_rate(prediction_rows)
        faith_rows = [
            r
            for r in prediction_rows
            if str(r.get("benchmark")) == "pubmedqa"
            and r.get("rag_context_used") in (True, 1, 1.0)
            and r.get("evidence_token_overlap") not in (None, "", float("nan"))
        ]
        if faith_rows:
            ovs = [float(r["evidence_token_overlap"]) for r in faith_rows]
            m, lo, hi = _bootstrap_ci(ovs)
            agg["faithfulness_proxy"] = {
                "metric": "evidence_token_overlap_jaccard",
                "n": len(ovs),
                "mean": m,
                "ci_lower": lo,
                "ci_upper": hi,
                "note": "Minimal in-core proxy (audit M10); not a clinical faithfulness claim.",
            }

    work = _work_dir()
    if os.path.isdir(work):
        agg["helper_bundle_hashes"] = _helper_bundle_hashes(work)

    if _env_truthy("GP_BENCH_STRICT_METRICS"):
        _enforce_strict_metrics(agg)

    out_dir = os.path.dirname(os.path.abspath(out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"\nWrote {out_json}", flush=True)
    return agg


def _default_rag_working_dir() -> str:
    if _on_colab():
        return "/content/rag_index"
    return os.path.join(_work_dir(), "rag_index")


# Google Colab + Drive (read in place when mounted).
COLAB_RAG_DATASET_DIR = "/content/drive/MyDrive/Colab Notebooks/rag_index"
COLAB_RAG_CHUNKS_JSONL = os.path.join(COLAB_RAG_DATASET_DIR, "chunks.jsonl")
COLAB_RAG_FAISS_INDEX = os.path.join(COLAB_RAG_DATASET_DIR, "index.faiss")

# Kaggle attached dataset (read in place when present).
# Primary: salmashopna/rag-index — all index files at dataset root.
KAGGLE_RAG_DATASET_DIR = "/kaggle/input/datasets/salmashopna/rag-index"
# PubMedQA recall@k gold: salmashopna/rag-index-gold (dual corpus with primary above).
KAGGLE_GOLD_RAG_DATASET_DIR = "/kaggle/input/datasets/salmashopna/rag-index-gold"
KAGGLE_GOLD_RAG_CHUNKS_JSONL = os.path.join(KAGGLE_GOLD_RAG_DATASET_DIR, "chunks.jsonl")
KAGGLE_GOLD_RAG_FAISS_INDEX = os.path.join(KAGGLE_GOLD_RAG_DATASET_DIR, "index.faiss")
KAGGLE_GOLD_RAG_MANIFEST_JSON = os.path.join(KAGGLE_GOLD_RAG_DATASET_DIR, "pubmedqa_gold_manifest.json")
KAGGLE_RAG_CHUNKS_JSONL = os.path.join(KAGGLE_RAG_DATASET_DIR, "chunks.jsonl")
KAGGLE_RAG_FAISS_INDEX = os.path.join(KAGGLE_RAG_DATASET_DIR, "index.faiss")
KAGGLE_RAG_CHUNKS_PAPER_556_JSONL = os.path.join(KAGGLE_RAG_DATASET_DIR, "chunks_paper_556.jsonl")
KAGGLE_RAG_FAISS_PAPER_556 = os.path.join(KAGGLE_RAG_DATASET_DIR, "index_paper_556.faiss")
KAGGLE_RAG_PAPER_CORPUS_STATS_JSON = os.path.join(KAGGLE_RAG_DATASET_DIR, "paper_corpus_stats.json")
KAGGLE_RAG_PAPER_SOURCE_MANIFEST_JSONL = os.path.join(KAGGLE_RAG_DATASET_DIR, "paper_source_manifest.jsonl")
# Alternate / legacy Kaggle layouts.
KAGGLE_RAG_DATASET_DIR_HAFIJUR = "/kaggle/input/datasets/hafijur222/rag-index"
KAGGLE_RAG_CHUNKS_JSONL_HAFIJUR = os.path.join(KAGGLE_RAG_DATASET_DIR_HAFIJUR, "chunks.jsonl")
KAGGLE_RAG_FAISS_INDEX_HAFIJUR = os.path.join(KAGGLE_RAG_DATASET_DIR_HAFIJUR, "index.faiss")
KAGGLE_RAG_DATASET_DIR_SIFATALI = "/kaggle/input/datasets/sifatali008/rag-index"
KAGGLE_RAG_CHUNKS_JSONL_SIFATALI = os.path.join(KAGGLE_RAG_DATASET_DIR_SIFATALI, "chunks.jsonl")
KAGGLE_RAG_FAISS_INDEX_SIFATALI = os.path.join(KAGGLE_RAG_DATASET_DIR_SIFATALI, "index.faiss")
KAGGLE_RAG_DATASET_DIR_SIDDARTH = os.path.join(
    "/kaggle/input/datasets/siddarthosarker/rag-index", "rag_index"
)
KAGGLE_RAG_CHUNKS_JSONL_SIDDARTH = os.path.join(KAGGLE_RAG_DATASET_DIR_SIDDARTH, "chunks.jsonl")
KAGGLE_RAG_FAISS_INDEX_SIDDARTH = os.path.join(KAGGLE_RAG_DATASET_DIR_SIDDARTH, "index.faiss")
KAGGLE_RAG_DATASET_DIR_LEGACY = "/kaggle/input/datasets/sifatali008/rag-index1"
KAGGLE_RAG_CHUNKS_JSONL_LEGACY = os.path.join(KAGGLE_RAG_DATASET_DIR_LEGACY, "chunks.jsonl")
KAGGLE_RAG_FAISS_INDEX_LEGACY = os.path.join(KAGGLE_RAG_DATASET_DIR_LEGACY, "index.faiss")
# Optional files copied alongside the main pair when staging to /kaggle/working/rag_index (if present at src).
KAGGLE_RAG_STAGE_SIDECAR_FILENAMES = (
    "chunks_paper_556.jsonl",
    "index_paper_556.faiss",
    "paper_corpus_stats.json",
    "paper_source_manifest.jsonl",
)
# All standard files under ``KAGGLE_RAG_DATASET_DIR`` when the full Kaggle dataset is attached.
KAGGLE_RAG_DATASET_ALL_KNOWN_PATHS: Tuple[str, ...] = (
    KAGGLE_RAG_CHUNKS_JSONL,
    KAGGLE_RAG_CHUNKS_PAPER_556_JSONL,
    KAGGLE_RAG_FAISS_INDEX,
    KAGGLE_RAG_FAISS_PAPER_556,
    KAGGLE_RAG_PAPER_CORPUS_STATS_JSON,
    KAGGLE_RAG_PAPER_SOURCE_MANIFEST_JSONL,
)


def _direct_rag_dir_candidates() -> List[str]:
    """Absolute index directories tried before generic search (Colab Drive, then Kaggle)."""
    dirs: List[str] = []
    extra = (os.environ.get("GP_RAG_DATASET_DIR") or "").strip()
    if extra:
        dirs.append(extra)
    sd0 = _script_dir()
    if sd0:
        dirs.append(os.path.join(sd0, "kaggle_working", "rag_index"))
    if _on_colab():
        sd = _script_dir()
        dirs.extend(
            [
                COLAB_RAG_DATASET_DIR,
                "/content/drive/MyDrive/rag_index",
                "/content/rag_index",
            ]
        )
        if sd:
            dirs.append(os.path.join(sd, "rag_index"))
            parent = os.path.dirname(sd)
            if parent:
                dirs.append(os.path.join(parent, "rag_index"))
    if os.path.isdir("/kaggle"):
        dirs.extend(
            [
                "/kaggle/working/rag_index",
                KAGGLE_RAG_DATASET_DIR,
                KAGGLE_RAG_DATASET_DIR_HAFIJUR,
                KAGGLE_RAG_DATASET_DIR_SIFATALI,
                KAGGLE_RAG_DATASET_DIR_SIDDARTH,
                KAGGLE_RAG_DATASET_DIR_LEGACY,
                "/kaggle/input/salmashopna/rag-index",
                "/kaggle/input/datasets/salmashopna/rag-index",
                "/kaggle/input/hafijur222/rag-index",
                "/kaggle/input/datasets/hafijur222/rag-index",
                "/kaggle/input/sifatali008/rag-index",
                "/kaggle/input/datasets/sifatali008/rag-index",
                "/kaggle/input/sifatali008/rag-index1",
                "/kaggle/input/siddarthosarker/rag-index/rag_index",
            ]
        )
    out: List[str] = []
    seen: set[str] = set()
    for d in dirs:
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            out.append(ad)
    return out


def _resolve_direct_rag() -> Optional[Tuple[str, str, str]]:
    """
    Return (index_dir, index.faiss path, chunks.jsonl path) for a known bundled index.

    Uses files in place (Colab Drive or Kaggle input) without copying to working dir.
    """
    for d in _direct_rag_dir_candidates():
        if _has_rag_pair(d):
            ip, cp = _rag_pair_in_dir(d)
            return d, ip, cp
    for base, ip, cp in (
        (COLAB_RAG_DATASET_DIR, COLAB_RAG_FAISS_INDEX, COLAB_RAG_CHUNKS_JSONL),
        (KAGGLE_RAG_DATASET_DIR, KAGGLE_RAG_FAISS_INDEX, KAGGLE_RAG_CHUNKS_JSONL),
        (KAGGLE_RAG_DATASET_DIR_HAFIJUR, KAGGLE_RAG_FAISS_INDEX_HAFIJUR, KAGGLE_RAG_CHUNKS_JSONL_HAFIJUR),
        (KAGGLE_RAG_DATASET_DIR_SIFATALI, KAGGLE_RAG_FAISS_INDEX_SIFATALI, KAGGLE_RAG_CHUNKS_JSONL_SIFATALI),
        (KAGGLE_RAG_DATASET_DIR_SIDDARTH, KAGGLE_RAG_FAISS_INDEX_SIDDARTH, KAGGLE_RAG_CHUNKS_JSONL_SIDDARTH),
        (KAGGLE_RAG_DATASET_DIR_LEGACY, KAGGLE_RAG_FAISS_INDEX_LEGACY, KAGGLE_RAG_CHUNKS_JSONL_LEGACY),
    ):
        if os.path.isfile(ip) and os.path.isfile(cp):
            return base, ip, cp
    return None


def _resolve_kaggle_direct_rag() -> Optional[Tuple[str, str, str]]:
    """Backward-compatible alias."""
    return _resolve_direct_rag()


def _kaggle_rag_subdir_names() -> Tuple[str, ...]:
    """Relative paths under each ``/kaggle/input/<dataset>`` root (first match wins)."""
    custom = (os.environ.get("GP_RAG_SUBDIR") or "").strip()
    names: List[str] = []
    if custom:
        names.append(custom)
    names.extend(
        [
            os.path.join("datasets", "salmashopna", "rag-index"),
            os.path.join("salmashopna", "rag-index"),
            os.path.join("datasets", "hafijur222", "rag-index"),
            os.path.join("hafijur222", "rag-index"),
            "rag-index",
            os.path.join("datasets", "sifatali008", "rag-index"),
            os.path.join("sifatali008", "rag-index"),
            os.path.join("datasets", "sifatali008", "rag-index1"),
            "rag-index1",
            os.path.join("datasets", "siddarthosarker", "rag-index", "rag_index"),
            os.path.join("siddarthosarker", "rag-index", "rag_index"),
            os.path.join("rag-index", "rag_index"),
            "rag_index",
        ]
    )
    out: List[str] = []
    seen: set[str] = set()
    for n in names:
        key = n.replace("\\", "/")
        if key not in seen:
            seen.add(key)
            out.append(n)
    return tuple(out)


def _probe_known_rag_subdirs(root: str) -> Optional[str]:
    """Fast path: ``.../rag_index1`` (or ``GP_RAG_SUBDIR``) before a full tree walk."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return None
    for sub in _kaggle_rag_subdir_names():
        d = os.path.join(root, sub)
        if _has_rag_pair(d):
            return os.path.abspath(d)
    return None


def _rag_pair_in_dir(dirpath: str) -> Tuple[str, str]:
    return os.path.join(dirpath, "index.faiss"), os.path.join(dirpath, "chunks.jsonl")


def _has_rag_pair(dirpath: str) -> bool:
    ip, cp = _rag_pair_in_dir(dirpath)
    return os.path.isfile(ip) and os.path.isfile(cp) and os.path.getsize(ip) > 64 and os.path.getsize(cp) > 32


def _repo_kaggle_working_rag_index() -> str:
    """``<repo>/kaggle_working/rag_index`` when this script lives under GreenPaper_Kaggle_Benchmarks."""
    sd = _script_dir()
    if not sd:
        return ""
    return os.path.join(sd, "kaggle_working", "rag_index")


def _paper_rag_file_pair(dirpath: str) -> Optional[Tuple[str, str]]:
    """IEEE Access 115-doc / 556-segment sidecar files in the same folder as the 25k index."""
    d = os.path.abspath(dirpath)
    cp = os.path.join(d, "chunks_paper_556.jsonl")
    ip = os.path.join(d, "index_paper_556.faiss")
    if os.path.isfile(cp) and os.path.isfile(ip) and os.path.getsize(ip) > 64:
        return ip, cp
    return None


def _apply_paper_corpus_pins(ns: Any) -> bool:
    """Pin env to ``chunks_paper_556.jsonl`` + ``index_paper_556.faiss`` if requested and found."""
    want = bool(getattr(ns, "rag_paper_corpus", False)) or _env_truthy("GP_RAG_PAPER_556")
    if not want:
        return False
    roots: List[str] = []
    if getattr(ns, "rag_index_dir", "").strip():
        roots.append(ns.rag_index_dir.strip())
    if os.path.isdir("/kaggle"):
        roots.extend(
            [
                KAGGLE_RAG_DATASET_DIR,
                KAGGLE_RAG_DATASET_DIR_SIDDARTH,
                KAGGLE_RAG_DATASET_DIR_LEGACY,
            ]
        )
    roots.append(_default_rag_working_dir())
    rk = _repo_kaggle_working_rag_index()
    if rk:
        roots.append(rk)
    # Kaggle working / input often same tree as sidecars
    if os.path.isdir("/kaggle/working"):
        roots.append("/kaggle/working/rag_index")
    seen: Set[str] = set()
    for d in roots:
        ad = os.path.abspath(d)
        if ad in seen or not ad:
            continue
        seen.add(ad)
        pair = _paper_rag_file_pair(ad)
        if pair:
            ip, cp = pair
            _pin_rag_files(ad, ip, cp)
            print(f"RAG: paper corpus (556) -> {cp!r}\n    index -> {ip!r}", flush=True)
            return True
    print(
        "RAG WARNING: --rag_paper_corpus / GP_RAG_PAPER_556 set but "
        "chunks_paper_556.jsonl + index_paper_556.faiss not found in search paths.",
        flush=True,
    )
    return False


def _rag_chunks_metadata(chunks_path: str) -> Tuple[int, bool, int]:
    """
    Return (n_lines, is_builtin_seed, pubmed_source_hits).
    Seed index = ~24 generic diabetes/guideline snippets from real_model_runner auto-build.
    """
    n_lines = 0
    pubmed_hits = 0
    first_text = ""
    with open(chunks_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            if "pubmedqa_" in line:
                pubmed_hits += 1
            if n_lines == 1:
                try:
                    first_text = str(json.loads(line).get("text") or "")[:300]
                except json.JSONDecodeError:
                    first_text = line[:300]
    is_seed = n_lines <= 30 and (
        "metformin is commonly recommended" in first_text.lower()
        or pubmed_hits == 0
    )
    if n_lines >= 100 or pubmed_hits >= 20:
        is_seed = False
    return n_lines, is_seed, pubmed_hits


def _rag_corpus_profile_line(chunks_path: str, n_lines: int, pubmed_hits: int) -> str:
    """One-line hint for logs (25k external vs PubMedQA-aligned vs small/paper)."""
    medq = pub_abs = gdl = 0
    cap = min(500, max(50, n_lines))
    try:
        with open(chunks_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= cap:
                    break
                if "medquad_" in line:
                    medq += 1
                if "pubmed_" in line and "pubmedqa_" not in line:
                    pub_abs += 1
                if "guideline_" in line:
                    gdl += 1
    except OSError:
        pass
    if 400 <= n_lines <= 650 and (gdl or pub_abs or medq):
        return "profile=paper/small (~556 segments expected for IEEE corpus sidecar use --rag_paper_corpus)"
    if n_lines >= 8000 and medq + pub_abs > 10 and pubmed_hits < 50:
        return (
            "profile=external_non_leaking (~25k MedQuad+PubMed abs typical). "
            "PubMedQA recall@k vs pubmedqa_* gold is N/A unless you use a gold-aligned index."
        )
    if pubmed_hits >= 50:
        return "profile=pubmedqa_aligned (pubmedqa_* in many rows — recall@k may be meaningful)"
    return "profile=mixed_or_custom"


def _walk_rag_index_dirs(root: str, max_depth: int = 4) -> List[str]:
    found: List[str] = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames.clear()
            continue
        if "index.faiss" in filenames and "chunks.jsonl" in filenames:
            found.append(dirpath)
    return found


def _discover_rag_index_dirs() -> List[Tuple[str, int, bool, int]]:
    """(dir, n_chunks, is_seed, pubmed_hits) — search Kaggle input before working."""
    out: List[Tuple[str, int, bool, int]] = []
    seen: set[str] = set()
    direct = _resolve_direct_rag()
    if direct:
        d, _ip, cp = direct
        if d not in seen:
            seen.add(d)
            n_lines, is_seed, pubmed_hits = _rag_chunks_metadata(cp)
            out.append((d, n_lines, is_seed, pubmed_hits))
    search_roots: List[str] = []
    if _on_colab():
        for root in (
            "/content/drive/MyDrive",
            "/content/drive/MyDrive/Colab Notebooks",
            "/content",
        ):
            if os.path.isdir(root) and root not in search_roots:
                search_roots.append(root)
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            search_roots.append(os.path.join(kin, name))
    for extra in (_script_dir(), _work_dir(), os.getcwd()):
        if extra and extra not in search_roots:
            search_roots.append(extra)
    for root in search_roots:
        hit = _probe_known_rag_subdirs(root)
        if hit and hit not in seen:
            seen.add(hit)
            cp = os.path.join(hit, "chunks.jsonl")
            n_lines, is_seed, pubmed_hits = _rag_chunks_metadata(cp)
            out.append((hit, n_lines, is_seed, pubmed_hits))
        for d in _walk_rag_index_dirs(root):
            ad = os.path.abspath(d)
            if ad in seen or not _has_rag_pair(ad):
                continue
            seen.add(ad)
            cp = os.path.join(ad, "chunks.jsonl")
            n_lines, is_seed, pubmed_hits = _rag_chunks_metadata(cp)
            out.append((ad, n_lines, is_seed, pubmed_hits))
    return out


def _rag_dir_name_priority(path: str) -> int:
    """Prefer Colab Drive and Kaggle ``salmashopna/rag-index`` (dataset root), then alternates."""
    low = path.replace("\\", "/").lower()
    if "colab notebooks/rag_index" in low or "colab notebooks\\rag_index" in low:
        return 7
    if "salmashopna" in low and "rag-index" in low and "rag-index-gold" not in low:
        return 8
    if "hafijur222" in low and "rag-index" in low and "rag-index1" not in low:
        return 7
    if "/datasets/sifatali008/rag-index" in low and "rag-index1" not in low:
        return 6
    if "siddarthosarker" in low and "rag-index" in low:
        return 6
    if "sifatali008" in low and "rag-index1" in low:
        return 5
    if "rag-index1" in low or "rag_index1" in low:
        return 4
    if low.endswith("/rag_index") or "/rag_index/" in low:
        return 2
    return 1


def _pick_best_rag_dir(candidates: List[Tuple[str, int, bool, int]]) -> Optional[str]:
    if not candidates:
        return None
    real = [c for c in candidates if not c[2]]
    pool = real if real else candidates
    pool.sort(key=lambda x: (-x[3], -x[1], -_rag_dir_name_priority(x[0]), x[2]))
    return pool[0][0]


def _stage_rag_index(src_dir: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    for name in ("index.faiss", "chunks.jsonl"):
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dest_dir, name))
    for name in KAGGLE_RAG_STAGE_SIDECAR_FILENAMES:
        sp = os.path.join(src_dir, name)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(dest_dir, name))
    return os.path.abspath(dest_dir)


def _pin_rag_environment(index_dir: str) -> str:
    """Set RAG_INDEX_DIR and explicit FAISS/chunks paths for real_model_runner."""
    d = os.path.abspath(index_dir.strip())
    return _pin_rag_files(d, os.path.join(d, "index.faiss"), os.path.join(d, "chunks.jsonl"))


def _pin_rag_files(index_dir: str, faiss_path: str, chunks_path: str) -> str:
    """Pin directory plus explicit index/chunks file paths (Kaggle input dataset)."""
    d = os.path.abspath(index_dir.strip())
    os.environ["RAG_INDEX_DIR"] = d
    os.environ["RAG_FAISS_INDEX"] = os.path.abspath(faiss_path)
    os.environ["RAG_CHUNKS_JSONL"] = os.path.abspath(chunks_path)
    if not os.environ.get("RAG_EMBED_MODEL", "").strip():
        os.environ["RAG_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["RAG_AUTO_BUILD"] = "0"
    return d


def _prewarm_rag_faiss() -> bool:
    _setup_runner_path()
    import real_model_runner as rmr

    rmr.ensure_rag_index_env()
    return bool(rmr.prewarm_faiss_index())


def _rag_env_snapshot() -> Dict[str, str]:
    keys = (
        "RAG_INDEX_DIR",
        "RAG_FAISS_INDEX",
        "RAG_CHUNKS_JSONL",
        "RAG_TOP_K",
        "RAG_CONTEXT_MAX_CHARS",
        "RAG_EMBED_MODEL",
        "RAG_FORCE_MOCK",
        "RAG_AUTO_BUILD",
    )
    return {k: os.environ.get(k, "") for k in keys}


def _print_rag_status() -> None:
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    if not d:
        print("RAG: no RAG_INDEX_DIR set.", flush=True)
        return
    cp = os.path.join(d, "chunks.jsonl")
    if not os.path.isfile(cp):
        print(f"RAG: RAG_INDEX_DIR={d!r} but chunks.jsonl missing.", flush=True)
        return
    n_lines, is_seed, pubmed_hits = _rag_chunks_metadata(cp)
    kind = "builtin seed (diabetes snippets)" if is_seed else "corpus"
    prof = _rag_corpus_profile_line(cp, n_lines, pubmed_hits)
    print(
        f"RAG: using {d!r} — {n_lines} chunks, {pubmed_hits} pubmedqa_* rows ({kind}). {prof}",
        flush=True,
    )


def _apply_rag_env_from_args(ns: Any) -> Dict[str, str]:
    """Apply --rag_* CLI flags to the environment; return a snapshot for results JSON."""
    used_paper = _apply_paper_corpus_pins(ns)
    if not used_paper and ns.rag_index_dir.strip():
        _pin_rag_environment(ns.rag_index_dir.strip())
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
    embed = (ns.rag_embed_model.strip() if hasattr(ns, "rag_embed_model") else "") or os.environ.get(
        "RAG_EMBED_MODEL", ""
    ).strip()
    if not embed:
        embed = "sentence-transformers/all-MiniLM-L6-v2"
    os.environ["RAG_EMBED_MODEL"] = embed
    return _rag_env_snapshot()


def _configure_rag_for_run(ns: Any) -> Dict[str, str]:
    """
    Apply CLI RAG flags, then on Kaggle use the attached salmashopna/rag-index dataset
    in place when present; otherwise stage the best index from /kaggle/input to working.
    """
    snapshot = _apply_rag_env_from_args(ns)
    if ns.rag_use_mock:
        print(
            "RAG: --rag_use_mock set; skipping FAISS (lexical retrieval over chunks.jsonl).",
            flush=True,
        )
        return snapshot

    user_set = bool(
        ns.rag_index_dir.strip() or ns.rag_faiss.strip() or ns.rag_chunks.strip()
    ) or bool(getattr(ns, "rag_paper_corpus", False)) or _env_truthy("GP_RAG_PAPER_556")
    if user_set:
        _print_rag_status()
        if not ns.rag_use_mock:
            os.environ["RAG_AUTO_BUILD"] = "0"
        return _rag_env_snapshot()

    direct = _resolve_direct_rag()
    working_dest = os.path.abspath(_default_rag_working_dir())
    if direct and not _env_truthy("GP_RAG_FORCE_STAGE"):
        d, ip, cp = direct
        _pin_rag_files(d, ip, cp)
        label = "Colab Drive" if _on_colab() and COLAB_RAG_DATASET_DIR in d else "attached dataset"
        print(
            f"RAG: using {label} directly (no staging):\n"
            f"  RAG_CHUNKS_JSONL={cp}\n"
            f"  RAG_FAISS_INDEX={ip}",
            flush=True,
        )
        _print_rag_status()
        return _rag_env_snapshot()

    if getattr(ns, "no_rag_stage", False):
        if direct:
            d, ip, cp = direct
            _pin_rag_files(d, ip, cp)
            print(
                "RAG: --no_rag_stage: using input dataset directly:\n"
                f"  {cp}\n  {ip}",
                flush=True,
            )
        _print_rag_status()
        return _rag_env_snapshot()

    dest = working_dest
    candidates = _discover_rag_index_dirs()
    best = _pick_best_rag_dir(candidates)

    if best:
        if os.path.abspath(best) != os.path.abspath(dest):
            _stage_rag_index(best, dest)
            print(f"RAG: staged index from {best!r} -> {dest!r}", flush=True)
        else:
            print(f"RAG: using index already at {dest!r}", flush=True)
        _pin_rag_environment(dest)
    elif _has_rag_pair(dest):
        n_lines, is_seed, pubmed_hits = _rag_chunks_metadata(os.path.join(dest, "chunks.jsonl"))
        _pin_rag_environment(dest)
        if is_seed:
            print(
                f"RAG WARNING: {dest!r} is the builtin seed index ({n_lines} chunks, "
                f"{pubmed_hits} pubmed rows). Attach your Kaggle dataset with real "
                "index.faiss + chunks.jsonl or pass --rag_index_dir.",
                flush=True,
            )
        else:
            os.environ["RAG_AUTO_BUILD"] = "0"
    else:
        hint = (
            f"Colab: mount Drive and place index under {COLAB_RAG_DATASET_DIR!r}, or set "
            f"GP_RAG_DATASET_DIR / --rag_index_dir. "
            if _on_colab()
            else (
                "Kaggle: attach salmashopna/rag-index or pass --rag_index_dir "
                "/kaggle/working/rag_index. "
            )
        )
        print(
            "RAG WARNING: no index.faiss + chunks.jsonl found "
            f"(looked for subfolders: {', '.join(_kaggle_rag_subdir_names())}). "
            f"{hint}"
            "A seed index may be auto-built.",
            flush=True,
        )

    _print_rag_status()
    snap = _rag_env_snapshot()
    try:
        _setup_runner_path()
        _drop_runner_caches()
        import real_model_runner as rmr

        rmr.ensure_rag_index_env()
        status = rmr.verify_rag_index(require_faiss=False)
        if status.get("index_path") and status.get("faiss_deps_ok") and status.get("faiss_load_ok"):
            print(
                f"RAG verify: FAISS OK — {status.get('ntotal')} vectors, d={status.get('dimension')}, "
                f"type={status.get('index_type')}, embed={status.get('embed_model')!r}",
                flush=True,
            )
            _prewarm_rag_faiss()
        elif status.get("index_path") and not status.get("faiss_deps_ok"):
            err = status.get("faiss_import_error") or "unknown"
            msg = (
                f"RAG index at {status.get('index_path')!r} but faiss-cpu missing ({err}). "
                "pip install faiss-cpu, restart kernel, re-run with !python eval_benchmarks.py."
            )
            if _kaggle_strict_checks():
                raise RuntimeError(msg)
            print(f"RAG WARNING: {msg}", flush=True)
    except RuntimeError:
        raise
    except Exception as ex:
        print(f"RAG verify skipped: {ex}", flush=True)
    return snap


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
        "--mmlu_split",
        default="dev",
        choices=["dev", "validation", "test", "train"],
        help="MMLU-Med split from cais/mmlu (default dev; use test only with pretraining caveat).",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=100,
        help="Cap per benchmark (default 100; use 500 for full paper runs; 10 for quick smoke; 0 = full HF split). "
        "Default: random sample of N per benchmark (new seed each run). Use --seed to fix the draw.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the random subset (same N questions on every run with this seed). "
        "If omitted, a new random seed is chosen and printed.",
    )
    p.add_argument(
        "--first_n",
        action="store_true",
        help="Take the first max_items in dataset order (no shuffle). Default: random sample.",
    )
    p.add_argument(
        "--save_predictions",
        default="",
        help="Path prefix for {prefix}_predictions.json and .csv. If omitted, defaults to the same "
        "directory and basename as --out_json (without .json), e.g. out_json .../foo.json -> "
        ".../foo_predictions.json. Use --no_save_predictions to disable.",
    )
    p.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Do not write *_predictions.json / .csv (overrides default auto-save).",
    )
    p.add_argument("--out_json", default="", help=f"default: {_default_out_json()}")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (ablation: less compression loss, more VRAM).")
    p.add_argument(
        "--rag_index_dir",
        default="",
        help="Directory with index.faiss + chunks.jsonl (~25k external corpus). "
        "If empty: RAG_INDEX_DIR, GP_RAG_DATASET_DIR, Kaggle input, or repo kaggle_working/rag_index.",
    )
    p.add_argument(
        "--rag_paper_corpus",
        action="store_true",
        help="Use chunks_paper_556.jsonl + index_paper_556.faiss in --rag_index_dir, "
        "repo kaggle_working/rag_index, or /kaggle/working/rag_index (IEEE Access 556 segments). "
        "Env: GP_RAG_PAPER_556=1.",
    )
    p.add_argument(
        "--no_rag_stage",
        action="store_true",
        help="Skip copy to /kaggle/working/rag_index (default when salmashopna/rag-index is attached).",
    )
    p.add_argument(
        "--rag_faiss",
        default="",
        help="Path to FAISS index file (overrides dir). Defaults: "
        f"Colab {COLAB_RAG_FAISS_INDEX} or Kaggle {KAGGLE_RAG_FAISS_INDEX}",
    )
    p.add_argument(
        "--rag_chunks",
        default="",
        help="Path to chunks JSONL (same row order as index). Defaults: "
        f"Colab {COLAB_RAG_CHUNKS_JSONL} or Kaggle {KAGGLE_RAG_CHUNKS_JSONL}",
    )
    p.add_argument("--rag_top_k", type=int, default=3, help="Chunks after rerank/dedup (default 3).")
    p.add_argument(
        "--rag_context_max_chars",
        type=int,
        default=2000,
        help="Max characters of retrieved evidence in the prompt (RAG repair plan: 1200–2500; default 2000).",
    )
    p.add_argument(
        "--rag_use_mock",
        action="store_true",
        help="Force mock RAG even if a FAISS index is configured (ablation baseline).",
    )
    p.add_argument(
        "--allow_lexical_rag",
        action="store_true",
        help="Allow lexical-only RAG when faiss-cpu is missing (invalidates recall@k ablations; not for paper). "
        "Hard-fails when PubMedQA is in the benchmark list (audit m5).",
    )
    p.add_argument(
        "--strict_metrics",
        action="store_true",
        help="Fail the run if primary metrics are NaN (audit m7). Also GP_BENCH_STRICT_METRICS=1.",
    )
    p.add_argument(
        "--no_faiss_autopip",
        action="store_true",
        help="Do not auto pip install faiss-cpu on Kaggle when import fails.",
    )
    p.add_argument(
        "--rag_embed_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model for query encoding (must match index build). Default: all-MiniLM-L6-v2.",
    )
    p.add_argument(
        "--build_gold_index",
        default="",
        metavar="OUT_DIR",
        help="Build PubMedQA gold chunks.jsonl + index.faiss (pubmedqa_<pubid>) and exit. "
        "Use with --auto_gold_pubmedqa or --rag_gold_index_dir for recall@k on external MCQ index.",
    )
    p.add_argument(
        "--no_build_gold_faiss",
        action="store_true",
        help="With --build_gold_index: only write chunks.jsonl (skip FAISS embed).",
    )
    p.add_argument(
        "--rag_gold_index_dir",
        default="",
        metavar="DIR",
        help="PubMedQA-only gold FAISS index (recall@k). MCQ keeps --rag_index_dir / Kaggle default.",
    )
    p.add_argument(
        "--auto_gold_pubmedqa",
        action="store_true",
        help="Explicitly request gold index pin (also auto on PubMedQA runs when salmashopna/rag-index-gold "
        "or kaggle_working/rag_index_gold is present).",
    )
    p.add_argument(
        "--build_fixed_chunks",
        nargs="?",
        const="auto",
        metavar="INPUT_JSONL",
        help="Split long abstract chunks into overlapping windows, write chunks_fixed.jsonl, then exit. "
        "Default input: RAG_CHUNKS_JSONL or rag_index_dir/chunks.jsonl.",
    )
    p.add_argument(
        "--fixed_chunks_out",
        default="",
        help="Output JSONL for --build_fixed_chunks (default: chunks_fixed.jsonl beside input).",
    )
    p.add_argument(
        "--fixed_chunk_words",
        type=int,
        default=300,
        help="Target chunk size in words (~256–384 tokens; RAG repair plan Step 01).",
    )
    p.add_argument(
        "--fixed_chunk_overlap",
        type=int,
        default=54,
        help="Overlap in words (~18%% of chunk_words).",
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
    p.add_argument(
        "--allow_pandas3",
        action="store_true",
        help="Skip Kaggle pandas 2.x guard (not recommended; results may be invalid).",
    )
    p.add_argument(
        "--hf_router",
        action="store_true",
        help="Use Hugging Face Inference Router (OpenAI-compatible API) with HF_TOKEN; no local GPU.",
    )
    p.add_argument(
        "--hf_router_base_url",
        default="",
        help="OpenAI API base URL (default https://router.huggingface.co/v1). Env: HF_ROUTER_BASE_URL.",
    )
    p.add_argument(
        "--hf_router_model_slm",
        default="",
        help="Router model id for SLM configs. Env: HF_ROUTER_MODEL_SLM. Default: same as LLM if unset.",
    )
    p.add_argument(
        "--hf_router_model_llm",
        default="",
        help="Router model id for LLM configs. Env: HF_ROUTER_MODEL_LLM. "
        f"Default: {_DEFAULT_HF_ROUTER_LLM!r}.",
    )
    p.add_argument(
        "--post_eval",
        action="store_true",
        help="After benchmarks, run run_paper_eval_suite (judge + faithfulness + paper_tables.json).",
    )
    p.add_argument(
        "--post_eval_judge",
        action="store_true",
        help="With --post_eval: run LLM-as-judge (Table 3). Default when --post_eval is set.",
    )
    p.add_argument(
        "--post_eval_no_judge",
        action="store_true",
        help="With --post_eval: skip LLM-as-judge.",
    )
    p.add_argument(
        "--post_eval_faithfulness",
        action="store_true",
        help="With --post_eval: run faithfulness scoring (Table 4). Default when --post_eval is set.",
    )
    p.add_argument(
        "--post_eval_no_faithfulness",
        action="store_true",
        help="With --post_eval: skip faithfulness.",
    )
    p.add_argument(
        "--post_eval_faithfulness_limit",
        type=int,
        default=0,
        help="Cap faithfulness rows (0=all RAG rows with context).",
    )
    p.add_argument(
        "--post_eval_faithfulness_backend",
        default="auto",
        choices=["auto", "openrouter", "deepeval"],
    )
    p.add_argument(
        "--post_eval_judge_max_rows",
        type=int,
        default=0,
        help="Cap LLM-judge rows (0=all).",
    )
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

    if getattr(args, "build_gold_index", "").strip():
        out_dir = args.build_gold_index.strip()
        embed = (args.rag_embed_model.strip() if hasattr(args, "rag_embed_model") else "") or ""
        build_pubmedqa_gold_rag_index(
            out_dir,
            max_items=int(args.max_items or 0),
            build_faiss=not getattr(args, "no_build_gold_faiss", False),
            embed_model=embed,
        )
        print(
            f"\nGold PubMedQA RAG index ready at {os.path.abspath(out_dir)!r}.\n"
            f"Dual corpus (external 25k + gold PubMedQA recall@k):\n"
            f"  python eval_benchmarks.py --benchmark all --auto_gold_pubmedqa --seed 42\n"
            f"Or full gold index for all benchmarks:\n"
            f"  python eval_benchmarks.py --benchmark all --rag_index_dir {out_dir!r} "
            f"--max_items 100 --seed 42\n",
            flush=True,
        )
        return

    if args.build_fixed_chunks is not None:
        inp = args.build_fixed_chunks
        if inp == "auto":
            if args.rag_chunks.strip():
                inp = os.path.abspath(args.rag_chunks.strip())
            elif args.rag_index_dir.strip():
                inp = os.path.join(os.path.abspath(args.rag_index_dir.strip()), "chunks.jsonl")
            else:
                inp = os.path.join(_default_rag_working_dir(), "chunks.jsonl")
        if not os.path.isfile(inp):
            print(f"chunks input not found: {inp}", file=sys.stderr)
            sys.exit(2)
        out = (args.fixed_chunks_out or "").strip() or os.path.join(
            os.path.dirname(os.path.abspath(inp)), "chunks_fixed.jsonl"
        )
        n = build_fixed_chunks(
            inp,
            out,
            chunk_words=args.fixed_chunk_words,
            overlap_words=args.fixed_chunk_overlap,
        )
        print(
            f"\nNext:\n"
            f"  python build_rag_index.py --input {out!r} --out_dir <rag_index_dir>\n"
            f"  !python eval_benchmarks.py --benchmark all --rag_index_dir <rag_index_dir> ...\n",
            flush=True,
        )
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

    _kaggle_prepare_environment()
    _warn_notebook_kernel_on_kaggle()
    _quiet_hf_datasets()
    if args.allow_pandas3:
        os.environ["GP_BENCH_ALLOW_PANDAS3"] = "1"
    if os.path.isdir("/kaggle"):
        print(
            "Kaggle: starting eval_benchmarks.py "
            f"(bundle {GP_BUNDLE_VERSION}, primary RAG {KAGGLE_RAG_DATASET_DIR}, "
            f"gold {KAGGLE_GOLD_RAG_DATASET_DIR})",
            flush=True,
        )
    _require_pandas2_on_kaggle(allow_pandas3=args.allow_pandas3)
    _require_numpy_scipy_stack()
    if not args.mock and not args.hf_router:
        _require_transformers_hf_stack()
    if not args.no_faiss_autopip:
        _ensure_faiss_cpu_installed()
    rag_snapshot = _configure_rag_for_run(args)
    gold_path = _apply_pubmedqa_gold_rag_from_args(args)
    if "pubmedqa" in benches:
        pinned = _pin_pubmedqa_gold_for_run(benches)
        if pinned:
            gold_path = pinned
    if gold_path:
        rag_snapshot["GP_RAG_GOLD_INDEX_DIR"] = gold_path
    if args.allow_lexical_rag and "pubmedqa" in benches:
        raise RuntimeError(
            "--allow_lexical_rag cannot be used with PubMedQA (audit m5): lexical ranking "
            "is not comparable to FAISS recall@k. Drop pubmedqa or install faiss-cpu."
        )
    if args.strict_metrics:
        os.environ["GP_BENCH_STRICT_METRICS"] = "1"
    _require_faiss_for_rag_run(
        allow_lexical=args.allow_lexical_rag,
        rag_use_mock=args.rag_use_mock,
        mock=args.mock,
        hf_router=args.hf_router,
        try_autopip=not args.no_faiss_autopip,
    )
    out_path = args.out_json or _default_out_json()
    save_pred = (args.save_predictions or "").strip()
    if args.no_save_predictions:
        save_pred = ""
    elif not save_pred:
        od = os.path.dirname(os.path.abspath(out_path)) or "."
        stem, _ = os.path.splitext(os.path.basename(out_path))
        if not stem:
            stem = "benchmark_results_all"
        save_pred = os.path.join(od, stem)
        pred_json = f"{save_pred}_predictions.json"
        print(f"Auto save_predictions: {pred_json}", flush=True)

    subset_seed = _resolve_subset_seed(args.max_items, args.seed, args.first_n)
    if args.max_items > 0 and not args.first_n:
        if args.seed is None:
            print(
                f"Random subset seed: {subset_seed} "
                f"(reproduce with --seed {subset_seed}; different each run if omitted)",
                flush=True,
            )
        else:
            print(f"Fixed subset seed: {subset_seed}", flush=True)
    elif args.max_items > 0 and args.first_n:
        print(f"Using first {args.max_items} items per benchmark (--first_n, no shuffle).", flush=True)

    os.environ["MMLU_MED_SPLIT"] = str(getattr(args, "mmlu_split", "dev") or "dev")

    run_all_benchmarks(
        benchmarks=benches,
        max_items=args.max_items,
        use_4bit=not args.no_4bit,
        data_paths=dmap,
        out_json=out_path,
        mock=args.mock,
        subset_seed=subset_seed,
        save_predictions_prefix=save_pred,
        rag_config=rag_snapshot,
        hf_router=args.hf_router,
        hf_router_base_url=args.hf_router_base_url,
        hf_router_model_slm=args.hf_router_model_slm,
        hf_router_model_llm=args.hf_router_model_llm,
    )

    if args.post_eval:
        pred_json = f"{save_pred}_predictions.json" if save_pred else ""
        if not pred_json or not os.path.isfile(pred_json):
            stem, _ = os.path.splitext(os.path.basename(out_path))
            od = os.path.dirname(os.path.abspath(out_path)) or "."
            pred_json = os.path.join(od, f"{stem or 'benchmark_results_all'}_predictions.json")
        if not os.path.isfile(pred_json):
            print(
                f"post_eval skipped: predictions not found at {pred_json!r} "
                "(run without --no_save_predictions).",
                flush=True,
            )
        else:
            if args.post_eval_judge or args.post_eval_faithfulness:
                run_judge = bool(args.post_eval_judge)
                run_faith = bool(args.post_eval_faithfulness)
            else:
                run_judge = not args.post_eval_no_judge
                run_faith = not args.post_eval_no_faithfulness
            try:
                from run_paper_eval_suite import run_post_eval

                tables_out = os.path.join(
                    os.path.dirname(os.path.abspath(out_path)) or ".",
                    "paper_tables.json",
                )
                print("\n=== Post-eval suite (faculty review plan) ===", flush=True)
                run_post_eval(
                    benchmark_json=out_path,
                    predictions_json=pred_json,
                    out_tables=tables_out,
                    run_judge=run_judge,
                    run_faithfulness=run_faith,
                    judge_max_rows=args.post_eval_judge_max_rows,
                    faithfulness_limit=args.post_eval_faithfulness_limit,
                    faithfulness_backend=args.post_eval_faithfulness_backend,
                )
            except Exception as ex:
                print(f"post_eval ERROR: {ex}", flush=True)
                raise


if __name__ == "__main__":
    # Notebook users sometimes paste only this block — imports live at top of file (~line 158).
    import os
    import sys

    if "main" not in globals():
        raise SystemExit(
            "eval_benchmarks.py: run the ENTIRE file in one notebook cell, not only the final "
            "`if __name__` block.\n"
            "  Paste the full script (optionally set sys.argv at the top); it auto-saves to "
            "/kaggle/working/eval_benchmarks.py on Kaggle.\n"
            "  Or: !python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 100 --seed 42"
        )

    if os.path.isdir("/kaggle"):
        try:
            _kaggle_prepare_environment()
        except Exception:
            pass
    _kaggle_reexec_if_notebook_kernel()
    main()