"""
LLM-as-judge (post-hoc) using a Hugging Face instruct model (e.g. Qwen) or HF Router.

**One Kaggle / Colab cell (copy whole file):** paste this file into a cell and run.
It auto-judges predictions from JSON or the merged CSV
``benchmark_results_all_predictions_combined.csv`` (multi-account exports).

**4-person split (rows 21,000+):**

- **Person 1**: global rows **21,000–26,339** (5,340 rows) — CSV: **fahim220**
- **Person 2**: global rows **26,340–31,679** (5,340 rows) — CSV: **fahim220**
- **Person 3**: global rows **31,680–end** (up to 5,340 rows) — CSV: **fatinshadab**
- **Person 4**: global rows **37,020–end** (only if needed) — CSV: **fatinshadab**

**Slice sifat28840 (default on Kaggle):** global rows **28,840–31,679** (2,840 rows) — CSV: **sifatali008/dataset**

**Slice hafijur30340:** global rows **30,340–31,679** (1,340 rows) — CSV: **hafijur222/dataset**

If your combined CSV has **36,360 rows**, then Person 3 is **31,680–36,359** (4,680 rows) and
there is **no Person 4**.

Fresh run — does **not** import old judge CSVs.

Outputs: ``/kaggle/working/LLM_Judge/<tag>/`` (e.g. ``rows28840_31679/``).

**Slice 28,840–31,679 Kaggle cell (default):**

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["LLM_JUDGE_SLICE"] = "sifat28840"
    os.environ["LLM_JUDGE_HF_ROUTER"] = "0"
    !python /kaggle/working/llm_judge.py --slice sifat28840 --batch_size 500

**Slice 30,340–31,679 Kaggle cell:**

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["LLM_JUDGE_SLICE"] = "hafijur30340"
    os.environ["LLM_JUDGE_HF_ROUTER"] = "0"
    !python /kaggle/working/llm_judge.py --slice hafijur30340 --batch_size 500

To run manually instead (or after ``LLM_JUDGE_AUTO=0`` to skip auto-run)::

    judge_predictions("", slice="sifat28840", batch_size=500)

**Kaggle batch workflow (36,360 rows):** each run judges **500 rows** and writes one CSV batch file.
Re-run the same cell until all batches finish (~73 runs). Progress is tracked via existing batch CSVs
under ``/kaggle/working/``. MCQ rows (MedQA, MMLU-Med, etc.) use letter-match-first scoring;
RAG context is treated as supplementary only.

Optional env overrides: ``LLM_JUDGE_BATCH_SIZE=500``, ``LLM_JUDGE_MAX_ROWS`` (0 = no random cap),
``LLM_JUDGE_SEED``, ``LLM_JUDGE_MODEL``,
``LLM_JUDGE_HF_ROUTER=1`` — use HF Router cloud API (uses Pro inference credits; not Kaggle GPU).
On Kaggle the default is **local GPU judge** (``hf_router=False``). Enable Router only if you set ``LLM_JUDGE_HF_ROUTER=1``.
``LLM_JUDGE_RUN_FOLDER=Bipro`` — only rows from one ``result/<folder>/`` export.
Set ``LLM_JUDGE_AUTO=0`` to define functions only without auto-run.

**Predictions inputs:**
- JSON: ``eval_benchmarks.py`` writes ``<out_json_stem>_predictions.json`` (default next to
  ``benchmark_results_all.json``).
- CSV: merged export ``result/benchmark_results_all_predictions_combined.csv`` with columns
  ``run_folder``, ``source_file``, plus the usual prediction fields.

**CLI (saved .py on disk):**

    python llm_judge.py --batch_size 500 --hf_router
    python llm_judge.py /path/to/benchmark_results_all_predictions_combined.csv --batch_size 500 --hf_router

**Batch outputs (Kaggle):** e.g.
``/kaggle/working/benchmark_results_all_predictions_combined_judge_batch_0001.csv`` (500 rows),
``..._batch_0002.csv``, … until ``..._batch_0073.csv`` (last partial batch).

**Checkpoint / resume:** mid-batch crashes resume from ``..._batch_NNNN.checkpoint.json``.
Completed batch CSVs are never re-judged.

Env: ``GP_JUDGE_MODEL``, ``HF_ROUTER_MODEL_JUDGE`` (router default; falls back to ``HF_ROUTER_MODEL_LLM``),
``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``, ``HF_ROUTER_BASE_URL``.
Enable Inference Providers at https://huggingface.co/settings/inference-providers .
If you hit HTTP 402 (credits depleted), add billing at https://huggingface.co/settings/billing
and re-run — the batch checkpoint keeps partial progress (e.g. 287/500 rows).
Or disable Router and use Kaggle GPU: ``LLM_JUDGE_HF_ROUTER=0`` then re-run.

**Resume after batch 0019:** (legacy full-run mode only) attach old batch CSVs via ``LLM_JUDGE_IMPORT_BATCH_DIR``.
Person-split runs ignore imported judge CSVs by default.
``LLM_JUDGE_NO_RESUME=1`` disables resume; ``LLM_JUDGE_CHECKPOINT_EVERY=N`` saves every N rows.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, List, Optional, Tuple

_SCORE_KEYS = ("correctness", "completeness", "clinical_relevance")
# Kaggle: one batch per run (500 rows → one CSV); re-run until all rows are judged.
_DEFAULT_BATCH_SIZE = 500
# Notebook auto-run: 0 = no random subsample (batch mode walks all rows in order).
_DEFAULT_NOTEBOOK_MAX_ROWS = 0

# Local GPU judge vs HF Router (must be enabled on your HF account).
_DEFAULT_JUDGE_MODEL_LOCAL = "Qwen/Qwen2.5-7B-Instruct"
# Same family as eval_benchmarks.py HF Router default; :fastest picks an enabled provider.
_DEFAULT_JUDGE_MODEL_ROUTER = "meta-llama/Llama-2-7b-chat-hf"
_ROUTER_JUDGE_FALLBACKS = (
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-3B-Instruct",
)
_ROUTER_MODEL_CACHE: Optional[str] = None


class RouterBillingError(RuntimeError):
    """HF Inference Provider credits exhausted (HTTP 402); checkpoint should be kept."""


def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


# 4-person split: each person judges 5,340 rows starting at global row 21,000.
_SPLIT_ROW_START = 21000
_SPLIT_ROWS_PER_PERSON = 5340
_PERSON_IDS = (1, 2, 3, 4)

# Completed judge batches (legacy local folder).
_DEFAULT_LLM_JUDGE_DIR = os.path.join(_script_dir(), "LLM_Judge")
_KAGGLE_JUDGE_WRITE_DIR = "/kaggle/working/LLM_Judge"
# Legacy import dirs (ignored when LLM_JUDGE_PERSON is set).
_KAGGLE_LLM_JUDGE_DIRS = (
    _KAGGLE_JUDGE_WRITE_DIR,
)
_KAGGLE_CSV_PERSON_12 = (
    "/kaggle/input/datasets/fahim220/benchmark-results-all-predictions-combined/"
    "benchmark_results_all_predictions_combined.csv"
)
_KAGGLE_CSV_PERSON_34 = (
    "/kaggle/input/datasets/fatinshadab/benchmark-results-all-predictions-combined/"
    "benchmark_results_all_predictions_combined.csv"
)
_KAGGLE_CSV_SIFATALI008 = (
    "/kaggle/input/datasets/sifatali008/dataset/"
    "benchmark_results_all_predictions_combined.csv"
)
_KAGGLE_CSV_HAFIJUR222 = (
    "/kaggle/input/datasets/hafijur222/dataset/"
    "benchmark_results_all_predictions_combined.csv"
)
# Global rows 28,840–31,679 (2,840 rows), 0-based slice [28840:31680).
_SLICE_SIFAT28840_OFFSET = 28840
_SLICE_SIFAT28840_LIMIT = 31680 - _SLICE_SIFAT28840_OFFSET
# Global rows 30,340–31,679 (1,340 rows), 0-based slice [30340:31680).
_SLICE_HAFIJUR30340_OFFSET = 30340
_SLICE_HAFIJUR30340_LIMIT = 31680 - _SLICE_HAFIJUR30340_OFFSET
_SLICE_PRESETS: Dict[str, Dict[str, Any]] = {
    "sifat28840": {
        "path": _KAGGLE_CSV_SIFATALI008,
        "row_offset": _SLICE_SIFAT28840_OFFSET,
        "row_limit": _SLICE_SIFAT28840_LIMIT,
        "tag": "rows28840_31679",
    },
    "hafijur30340": {
        "path": _KAGGLE_CSV_HAFIJUR222,
        "row_offset": _SLICE_HAFIJUR30340_OFFSET,
        "row_limit": _SLICE_HAFIJUR30340_LIMIT,
        "tag": "rows30340_31679",
    },
}


def _predictions_csv_for_person(person_id: int) -> str:
    """Persons 1–2 use fahim220; persons 3–4 (rows after 31,679) use fatinshadab."""
    if person_id in (1, 2):
        return _KAGGLE_CSV_PERSON_12
    if person_id in (3, 4):
        return _KAGGLE_CSV_PERSON_34
    return _KAGGLE_CSV_PERSON_34


# Default paths: merged multi-run CSV first, then single-run JSON from eval_benchmarks.
_DEFAULT_PREDICTIONS_CANDIDATES = (
    _KAGGLE_CSV_HAFIJUR222,
    _KAGGLE_CSV_SIFATALI008,
    _KAGGLE_CSV_PERSON_34,
    _KAGGLE_CSV_PERSON_12,
    "/kaggle/working/benchmark_results_all_predictions_combined.csv",
    "/kaggle/input/benchmark_results_all_predictions_combined.csv",
    os.path.join(os.getcwd(), "benchmark_results_all_predictions_combined.csv"),
    os.path.join(os.getcwd(), "result", "benchmark_results_all_predictions_combined.csv"),
    "/kaggle/working/benchmark_results_all_predictions.json",
    "benchmark_results_all_predictions.json",
    os.path.join(os.getcwd(), "benchmark_results_all_predictions.json"),
)


def _sort_rows_deterministic(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("run_folder") or ""),
            str(r.get("source_file") or ""),
            str(r.get("benchmark") or ""),
            str(r.get("question_id") or ""),
            str(r.get("model_name") or ""),
        ),
    )


def _person_tag(person_id: int) -> str:
    return f"person{person_id}" if person_id > 0 else ""


def _default_slice_name() -> str:
    """Default Kaggle slice: rows 28,840–31,679 on sifatali008 CSV."""
    env = os.environ.get("LLM_JUDGE_SLICE", "").strip()
    if env.lower() in ("0", "false", "no", "none"):
        return ""
    if env:
        return env
    if os.path.isdir("/kaggle"):
        p = os.environ.get("LLM_JUDGE_PERSON", "").strip()
        if p and p not in ("0", ""):
            return ""
        if os.environ.get("LLM_JUDGE_ROW_OFFSET", "").strip().isdigit():
            return ""
        return "sifat28840"
    return ""


def _resolve_active_slice(slice_name: str = "") -> Optional[Dict[str, Any]]:
    name = (slice_name or _default_slice_name()).strip()
    if not name:
        return None
    if name in _SLICE_PRESETS:
        return dict(_SLICE_PRESETS[name])
    raise ValueError(
        f"Unknown slice {name!r}. Available: {sorted(_SLICE_PRESETS.keys())}"
    )


def _person_slice(n_total: int, person_id: int) -> Tuple[int, int, str]:
    """Return (row_offset, row_limit, person_tag) for a team member."""
    if person_id not in _PERSON_IDS:
        raise ValueError(f"person must be one of {_PERSON_IDS}, got {person_id!r}")
    row_offset = _SPLIT_ROW_START + (person_id - 1) * _SPLIT_ROWS_PER_PERSON
    if row_offset >= n_total:
        raise ValueError(
            f"Person {person_id} starts at row {row_offset} but CSV only has {n_total} rows."
        )
    row_limit = min(_SPLIT_ROWS_PER_PERSON, n_total - row_offset)
    return row_offset, row_limit, _person_tag(person_id)


def _should_use_slice_preset(
    slice_name: str, person: int, row_offset: int, row_limit: int
) -> bool:
    if slice_name.strip() or os.environ.get("LLM_JUDGE_SLICE", "").strip():
        return True
    if person > 0 or row_offset > 0 or row_limit > 0:
        return False
    return bool(_default_slice_name())


def _resolve_row_slice(
    n_total: int,
    *,
    person: int = 0,
    row_offset: int = 0,
    row_limit: int = 0,
) -> Tuple[int, int, str, bool]:
    """
    Resolve row window. Returns (offset, limit, person_tag, person_mode).

    ``person_mode`` True → only resume from this person's output folder (no legacy imports).
    """
    env_person = os.environ.get("LLM_JUDGE_PERSON", "").strip()
    if person <= 0 and env_person.isdigit():
        person = int(env_person)
    if person > 0:
        off, lim, tag = _person_slice(n_total, person)
        return off, lim, tag, True
    if row_offset > 0 or row_limit > 0:
        off = max(0, row_offset)
        lim = row_limit if row_limit > 0 else max(0, n_total - off)
        tag = f"rows{off}_{off + lim - 1}" if lim > 0 else f"rows{off}_end"
        return off, lim, tag, True
    env_off = os.environ.get("LLM_JUDGE_ROW_OFFSET", "").strip()
    env_lim = os.environ.get("LLM_JUDGE_ROW_LIMIT", "").strip()
    if env_off.isdigit() or env_lim.isdigit():
        off = int(env_off) if env_off.isdigit() else 0
        lim = int(env_lim) if env_lim.isdigit() else max(0, n_total - off)
        tag = f"rows{off}_{off + lim - 1}" if lim > 0 else f"rows{off}_end"
        return off, lim, tag, True
    return 0, 0, "", False


def _apply_row_slice(
    rows: List[Dict[str, Any]], row_offset: int, row_limit: int
) -> List[Dict[str, Any]]:
    if row_offset <= 0 and row_limit <= 0:
        return rows
    end = len(rows) if row_limit <= 0 else min(len(rows), row_offset + row_limit)
    sliced = rows[row_offset:end]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(sliced):
        rec = dict(row)
        rec["global_row_index"] = str(row_offset + i)
        out.append(rec)
    return out


def _known_llm_judge_dirs(judge_stem_name: str, *, person_mode: bool = False) -> List[str]:
    """Folders scanned for completed batch CSVs."""
    if person_mode:
        env_out = os.environ.get("LLM_JUDGE_OUT_DIR", "").strip()
        if env_out:
            return [os.path.abspath(env_out)]
        tag = os.environ.get("LLM_JUDGE_OUTPUT_TAG", "").strip()
        if tag:
            return [_person_output_dir(tag)]
        person = os.environ.get("LLM_JUDGE_PERSON", "").strip()
        if person.isdigit() and int(person) > 0:
            return [_person_output_dir(_person_tag(int(person)))]
        return [_KAGGLE_JUDGE_WRITE_DIR if os.path.isdir("/kaggle") else _DEFAULT_LLM_JUDGE_DIR]

    candidates: List[str] = []
    env_out = os.environ.get("LLM_JUDGE_OUT_DIR", "").strip()
    env_imp = os.environ.get("LLM_JUDGE_IMPORT_BATCH_DIR", "").strip()
    if env_out:
        candidates.append(env_out)
    if env_imp:
        candidates.append(env_imp)
    raw = os.environ.get("LLM_JUDGE_IMPORT_BATCH_DIRS", "").strip()
    if raw:
        candidates.extend(x.strip() for x in raw.split(",") if x.strip())
    candidates.append(_DEFAULT_LLM_JUDGE_DIR)
    for base in (
        _script_dir(),
        os.getcwd(),
        os.path.join(os.getcwd(), "GreenPaper_Kaggle_Benchmarks"),
        os.path.dirname(_script_dir()),
    ):
        candidates.append(os.path.join(base, "LLM_Judge"))
    candidates.extend(_KAGGLE_LLM_JUDGE_DIRS)
    if os.environ.get("LLM_JUDGE_IMPORT_LEGACY_BATCHES", "").strip().lower() in ("1", "true", "yes"):
        candidates.append("/kaggle/input/datasets/sifatali008/output")
    # Do not auto-scan /kaggle/input for old batch_0000.csv (Person 1 uses a fresh folder).
    seen: set[str] = set()
    out: List[str] = []
    for d in candidates:
        if not d:
            continue
        ap = os.path.abspath(d)
        if ap in seen or not os.path.isdir(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _person_output_dir(person_tag: str) -> str:
    if person_tag:
        if os.path.isdir("/kaggle"):
            return os.path.join(_KAGGLE_JUDGE_WRITE_DIR, person_tag)
        return os.path.join(_DEFAULT_LLM_JUDGE_DIR, person_tag)
    if os.path.isdir("/kaggle"):
        return _KAGGLE_JUDGE_WRITE_DIR
    return _DEFAULT_LLM_JUDGE_DIR


def _known_llm_judge_dirs_legacy(judge_stem_name: str) -> List[str]:
    return _known_llm_judge_dirs(judge_stem_name, person_mode=False)


def _count_batch_csvs_in_dir(directory: str, judge_stem_name: str) -> int:
    pattern = os.path.join(directory, f"{judge_stem_name}_batch_*.csv")
    return len(glob.glob(pattern))


def _default_person() -> int:
    """Person split default (0 when slice mode is active on Kaggle)."""
    env = os.environ.get("LLM_JUDGE_PERSON", "").strip()
    if env.lower() in ("0", "false", "no", "none", ""):
        if env == "" and os.path.isdir("/kaggle"):
            return 0
        if env.lower() in ("0", "false", "no", "none"):
            return 0
    if env.isdigit():
        return int(env)
    return 0


def _resolve_judge_write_dir(judge_stem_name: str, person_tag: str = "") -> str:
    """
    Writable folder for new batch CSVs and checkpoints.

    Person-split runs write only under ``LLM_Judge/personN/`` (never legacy import dirs).
    """
    env_out = os.environ.get("LLM_JUDGE_OUT_DIR", "").strip()
    if env_out:
        os.makedirs(env_out, exist_ok=True)
        return os.path.abspath(env_out)
    if person_tag:
        d = _person_output_dir(person_tag)
        os.makedirs(d, exist_ok=True)
        return os.path.abspath(d)
    if os.path.isdir("/kaggle"):
        os.makedirs(_KAGGLE_JUDGE_WRITE_DIR, exist_ok=True)
        return os.path.abspath(_KAGGLE_JUDGE_WRITE_DIR)

    best_dir = ""
    best_n = -1
    for d in _known_llm_judge_dirs(judge_stem_name, person_mode=False):
        if not os.access(d, os.W_OK):
            continue
        n = _count_batch_csvs_in_dir(d, judge_stem_name)
        if n > best_n:
            best_n = n
            best_dir = d
    if best_dir:
        return best_dir

    os.makedirs(_DEFAULT_LLM_JUDGE_DIR, exist_ok=True)
    return os.path.abspath(_DEFAULT_LLM_JUDGE_DIR)


def _judge_output_stem(predictions_path: str, out_path: str, person_tag: str = "") -> str:
    if out_path.strip():
        return os.path.splitext(out_path.strip())[0]
    base = os.path.basename(predictions_path)
    stem_name = f"{os.path.splitext(base)[0]}_judge"
    out_dir = _resolve_judge_write_dir(stem_name, person_tag=person_tag)
    return os.path.join(out_dir, stem_name)


def _batch_filename(output_stem: str, batch_index: int) -> str:
    return f"{os.path.basename(output_stem)}_batch_{batch_index:04d}.csv"


def _batch_csv_path(output_stem: str, batch_index: int) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(output_stem)) or ".",
        _batch_filename(output_stem, batch_index),
    )


def _batch_search_dirs(output_stem: str, *, person_mode: bool = False) -> List[str]:
    """Directories to scan for completed batch CSVs (person mode: output folder only)."""
    primary = os.path.dirname(os.path.abspath(output_stem)) or "."
    judge_base = os.path.basename(output_stem)
    if person_mode:
        return [primary] if os.path.isdir(primary) else []
    extras = _known_llm_judge_dirs(judge_base, person_mode=False)
    seen: set[str] = set()
    out: List[str] = []
    for d in [primary] + extras:
        if not d:
            continue
        ap = os.path.abspath(d)
        if ap in seen or not os.path.isdir(ap):
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _find_existing_batch_csv(output_stem: str, batch_index: int, *, person_mode: bool = False) -> Optional[str]:
    fname = _batch_filename(output_stem, batch_index)
    for d in _batch_search_dirs(output_stem, person_mode=person_mode):
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None


def _batch_is_complete(
    output_stem: str, batch_index: int, expected_rows: int, *, person_mode: bool = False
) -> bool:
    path = _find_existing_batch_csv(output_stem, batch_index, person_mode=person_mode)
    if not path:
        return False
    return _count_csv_data_rows(path) >= expected_rows


def _count_completed_judge_rows(
    output_stem: str, n_batches: int, batch_sizes: List[int], *, person_mode: bool = False
) -> int:
    total = 0
    for idx in range(n_batches):
        path = _find_existing_batch_csv(output_stem, idx, person_mode=person_mode)
        if path and _count_csv_data_rows(path) >= batch_sizes[idx]:
            total += batch_sizes[idx]
    return total


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


def _last_complete_batch_index(
    output_stem: str, n_batches: int, batch_sizes: List[int], *, person_mode: bool = False
) -> Optional[int]:
    last: Optional[int] = None
    for idx in range(n_batches):
        if _batch_is_complete(output_stem, idx, batch_sizes[idx], person_mode=person_mode):
            last = idx
    return last


def _find_next_batch_index(
    output_stem: str, n_batches: int, batch_sizes: List[int], *, person_mode: bool = False
) -> Optional[int]:
    """Return the first batch index without a complete CSV in any search directory."""
    for idx in range(n_batches):
        if _batch_is_complete(output_stem, idx, batch_sizes[idx], person_mode=person_mode):
            continue
        return idx
    return None


def _row_to_csv_dict(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k == "judge_scores":
            continue
        out[k] = "" if v is None else str(v)
    scores = row.get("judge_scores")
    if isinstance(scores, dict):
        out["judge_correctness"] = str(scores.get("correctness") or "")
        out["judge_completeness"] = str(scores.get("completeness") or "")
        out["judge_clinical_relevance"] = str(scores.get("clinical_relevance") or "")
        out["judge_brief_rationale"] = str(scores.get("brief_rationale") or "")
        out["judge_parse_ok"] = "1"
    else:
        out["judge_correctness"] = ""
        out["judge_completeness"] = ""
        out["judge_clinical_relevance"] = ""
        out["judge_brief_rationale"] = ""
        out["judge_parse_ok"] = "0"
    out["judge_model"] = str(row.get("judge_model") or "")
    out["judge_raw"] = str(row.get("judge_raw") or "")
    return out


def _write_judge_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_rows = [_row_to_csv_dict(r) for r in rows]
    fieldnames: List[str] = []
    seen: set[str] = set()
    for d in csv_rows:
        for k in d:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    od = os.path.dirname(os.path.abspath(csv_path))
    if od:
        os.makedirs(od, exist_ok=True)
    tmp = csv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(tmp, csv_path)


def _split_batches(rows: List[Dict[str, Any]], batch_size: int) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    batches: List[List[Dict[str, Any]]] = []
    sizes: List[int] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        batches.append(chunk)
        sizes.append(len(chunk))
    return batches, sizes


def _is_csv_path(path: str) -> bool:
    return path.lower().endswith(".csv")


def _resolve_hf_router_flag(cli_hf_router: bool = False) -> bool:
    """
    Return whether to use HF Router (cloud API).

    Kaggle default is **False** — uses the notebook GPU via transformers.
    Set ``LLM_JUDGE_HF_ROUTER=1`` or ``--hf_router`` to use paid Router API instead.
    """
    env = os.environ.get("LLM_JUDGE_HF_ROUTER", "").strip().lower()
    if env in ("0", "false", "no", "n"):
        return False
    if env in ("1", "true", "yes", "y") or cli_hf_router:
        return True
    if os.path.isdir("/kaggle"):
        return False
    return False


def _log_judge_backend(hf_router: bool) -> None:
    if hf_router:
        print(
            "[llm_judge] Backend: HF Router (cloud API — bills HF Inference credits, not Kaggle GPU).",
            flush=True,
        )
        return
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(f"[llm_judge] Backend: local judge on Kaggle GPU (cuda={cuda}, device={dev}).", flush=True)
    except Exception:
        print("[llm_judge] Backend: local judge (transformers on CPU/GPU).", flush=True)


def _discover_kaggle_input_predictions() -> List[str]:
    """Scan ``/kaggle/input`` for combined CSV or ``*_predictions.json``."""
    found: List[str] = []
    kin = "/kaggle/input"
    if not os.path.isdir(kin):
        return found
    for root, _dirs, files in os.walk(kin):
        for name in files:
            low = name.lower()
            if low == "benchmark_results_all_predictions_combined.csv":
                found.append(os.path.join(root, name))
            elif low.endswith("_predictions.json") or name == "benchmark_results_all_predictions.json":
                found.append(os.path.join(root, name))
    return sorted(set(found))


def _load_predictions_csv(csv_path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    meta = {
        "source_format": "csv",
        "predictions_path": os.path.abspath(csv_path),
        "n_rows": len(rows),
    }
    folders = sorted({str(r.get("run_folder") or "") for r in rows if r.get("run_folder")})
    if folders:
        meta["run_folders"] = folders
    return {"meta": meta, "rows": rows}


def _load_predictions_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = list(payload.get("rows") or [])
    meta = dict(payload.get("meta") or {})
    meta["source_format"] = "json"
    meta["predictions_path"] = os.path.abspath(json_path)
    meta["n_rows"] = len(rows)
    return {"meta": meta, "rows": rows}


def _load_predictions_payload(path: str) -> Dict[str, Any]:
    if _is_csv_path(path):
        return _load_predictions_csv(path)
    return _load_predictions_json(path)


def _resolve_person_id(person: int = 0) -> int:
    if person > 0:
        return person
    env = os.environ.get("LLM_JUDGE_PERSON", "").strip()
    if env.isdigit():
        return int(env)
    return _default_person()


def _resolve_predictions_path(path: str, person: int = 0) -> str:
    """
    Return an existing predictions JSON or CSV path.

    If ``path`` is empty, pick CSV by person: 1–2 → fahim220, 3–4 → fatinshadab.
    """
    p = (path or "").strip()
    if p and os.path.isfile(p):
        return os.path.abspath(p)
    pid = _resolve_person_id(person)
    candidates: List[str] = []
    if p:
        candidates.append(p)
    candidates.append(_predictions_csv_for_person(pid))
    candidates.extend(_DEFAULT_PREDICTIONS_CANDIDATES)
    candidates.extend(_discover_kaggle_input_predictions())
    seen: set[str] = set()
    for c in candidates:
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            if p and c != os.path.abspath(p):
                print(
                    f"[llm_judge] {p!r} not found; using {c!r} instead.",
                    flush=True,
                )
            return c
    tried = ", ".join(repr(x) for x in candidates[:6])
    raise FileNotFoundError(
        f"Predictions file not found. Tried: {tried}. "
        "Attach fahim220 (person 1–2) or fatinshadab (person 3–4) predictions dataset on Kaggle. "
        "or run eval_benchmarks.py (writes benchmark_results_all_predictions.json). "
        f"Person 3–4 CSV: {_KAGGLE_CSV_PERSON_34!r}"
    )


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
                    print(f"[llm_judge] Loaded HF_TOKEN from Kaggle Secrets ({name}).", flush=True)
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


def _normalize_mcq_letter(value: str) -> str:
    """Extract a single MCQ option letter (A–Z) from a short answer string."""
    v = (value or "").strip().upper()
    if not v:
        return ""
    if len(v) == 1 and v.isalpha():
        return v
    m = re.match(r"^([A-Z])[\).:\-\s]", v)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z])\b", v)
    if m:
        return m.group(1)
    return ""


def _parse_choices_list(row: Dict[str, Any]) -> List[str]:
    raw = row.get("choices_json") or row.get("choices") or ""
    if not raw or not str(raw).strip():
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _is_letter_mcq_row(row: Dict[str, Any]) -> bool:
    ref = _normalize_mcq_letter(str(row.get("reference_answer") or row.get("mcq_correct") or ""))
    if ref:
        return True
    choices = _parse_choices_list(row)
    return len(choices) >= 2


def _extract_brief_rationale(obj: Dict[str, Any]) -> str:
    """Pull rationale text; tolerate common key typos from judge models."""
    for k in (
        "brief_rationale",
        "brief_rationationale",
        "rationationale",
        "rationale",
        "brief_reason",
        "reasoning",
        "reason",
        "explanation",
        "brief_explanation",
    ):
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()[:500]
    for k, v in obj.items():
        kl = str(k).lower()
        if "ration" in kl or kl in ("reason", "reasoning", "explanation"):
            if v is not None and str(v).strip():
                return str(v).strip()[:500]
    return ""


def _build_user_message(row: Dict[str, Any], max_ctx_chars: int) -> str:
    q = str(row.get("question") or "")
    ref_raw = str(row.get("reference_answer") or row.get("mcq_correct") or "")
    pred_raw = str(row.get("model_answer") or row.get("parsed_prediction") or row.get("raw_response") or "")
    ctx = _truncate(str(row.get("context") or ""), max_ctx_chars)
    rag = _truncate(str(row.get("retrieved_context") or ""), max_ctx_chars)
    bench = str(row.get("benchmark") or "")
    is_mcq = _is_letter_mcq_row(row)
    ref_letter = _normalize_mcq_letter(ref_raw)
    pred_letter = _normalize_mcq_letter(pred_raw)
    choices = _parse_choices_list(row)

    lines = [
        "You are an expert clinical evaluator scoring a model answer against the gold reference.",
        "Use integer scores from 1 (poor) to 5 (excellent) for each rubric dimension.",
    ]

    if is_mcq and ref_letter:
        lines.extend(
            [
                "",
                "This is a multiple-choice question (MCQ). Scoring rules (follow strictly):",
                "1. CORRECTNESS is driven primarily by whether the model's selected option LETTER matches the gold letter.",
                "   - Gold letter and model letter match → correctness must be 4 or 5 (never 1–3).",
                "   - Letters differ → correctness must be 1 or 2 (never 4–5), even if the rationale sounds plausible.",
                "2. Do NOT penalize a correct letter because retrieved RAG evidence is missing, irrelevant, or off-topic.",
                "3. Do NOT award high correctness when the letter is wrong, even if clinical reasoning sounds convincing.",
                "4. COMPLETENESS: quality/clarity of explanation (letter-only answers without reasoning → 3–4 if letter correct).",
                "5. CLINICAL_RELEVANCE: medical relevance of the chosen option and reasoning to the question stem.",
            ]
        )

    lines.extend(["", f"Benchmark: {bench}", f"Question: {q}"])

    if is_mcq and ref_letter:
        lines.append(f"Gold option letter: {ref_letter}")
        if ref_raw and ref_raw.strip().upper() != ref_letter:
            lines.append(f"Gold reference (full): {ref_raw}")
        if pred_letter:
            match_note = "MATCH" if pred_letter == ref_letter else "MISMATCH"
            lines.append(f"Model selected letter: {pred_letter} ({match_note} vs gold {ref_letter})")
        else:
            lines.append(f"Model selected letter: (could not parse — inspect model answer text below)")
        if choices:
            lines.append("")
            lines.append("Answer choices:")
            for i, choice in enumerate(choices):
                letter = chr(ord("A") + i)
                lines.append(f"  {letter}. {choice}")
    else:
        lines.append(f"Reference (gold): {ref_raw}")

    if ctx:
        lines.extend(["", f"Dataset context (if any):\n{ctx}"])
    if rag:
        lines.extend(
            [
                "",
                "Retrieved RAG evidence (supplementary only; may be irrelevant — do not override MCQ letter matching):",
                rag,
            ]
        )

    lines.extend(["", f"Model answer (full text):\n{pred_raw}"])

    lines.extend(
        [
            "",
            "Return ONLY valid JSON with this exact shape (no markdown fences):",
            '{"correctness": <1-5>, "completeness": <1-5>, "clinical_relevance": <1-5>, '
            '"brief_rationale": "<one sentence>"}',
        ]
    )
    return "\n".join(lines)


def _parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        lo, hi = text.find("{"), text.rfind("}")
        if lo == -1 or hi <= lo:
            return None
        try:
            obj = json.loads(text[lo : hi + 1])
        except json.JSONDecodeError:
            return None
    out: Dict[str, Any] = {}
    for k in _SCORE_KEYS:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            out[k] = int(round(max(1, min(5, float(v)))))
        elif isinstance(v, str) and v.strip().isdigit():
            out[k] = int(max(1, min(5, int(v.strip()))))
        else:
            return None
    out["brief_rationale"] = _extract_brief_rationale(obj)
    return out


def _normalize_router_model_id(model_id: str) -> str:
    """Append ``:fastest`` so HF Router picks an enabled provider automatically."""
    m = (model_id or "").strip()
    if not m or ":" in m:
        return m
    return f"{m}:fastest"


def _resolve_judge_model(model_id: str, hf_router: bool) -> str:
    mid = (model_id or "").strip()
    if not mid:
        mid = (os.environ.get("GP_JUDGE_MODEL") or "").strip()
    if not mid:
        if hf_router:
            mid = (
                os.environ.get("HF_ROUTER_MODEL_JUDGE")
                or os.environ.get("HF_ROUTER_MODEL_LLM")
                or _DEFAULT_JUDGE_MODEL_ROUTER
            ).strip()
        else:
            mid = _DEFAULT_JUDGE_MODEL_LOCAL
    if hf_router:
        mid = _normalize_router_model_id(mid)
    return mid


def _router_judge_model_candidates(primary: str) -> List[str]:
    env = os.environ.get("HF_ROUTER_JUDGE_FALLBACKS", "").strip()
    fallbacks = [x.strip() for x in env.split(",") if x.strip()] if env else list(_ROUTER_JUDGE_FALLBACKS)
    seen: set[str] = set()
    out: List[str] = []
    for m in [primary] + fallbacks:
        m = _normalize_router_model_id(m)
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _http_status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _is_router_billing_error(exc: BaseException) -> bool:
    if _http_status_code(exc) == 402:
        return True
    msg = str(exc).lower()
    needles = (
        "depleted your monthly",
        "pre-paid credits",
        "prepaid credits",
        "payment required",
        "insufficient credits",
        "out of credits",
        "purchase credits",
    )
    return any(n in msg for n in needles)


def _is_router_transient_error(exc: BaseException) -> bool:
    sc = _http_status_code(exc)
    if sc in (408, 429, 500, 502, 503, 504):
        return True
    msg = str(exc).lower()
    return any(
        n in msg
        for n in ("rate limit", "timeout", "timed out", "overloaded", "temporarily unavailable")
    )


def _is_router_model_unsupported(exc: BaseException) -> bool:
    if _is_router_billing_error(exc):
        return False
    msg = str(exc).lower()
    if "model_not_supported" in msg or "not supported by any provider" in msg:
        return True
    code = getattr(exc, "code", None) or getattr(getattr(exc, "body", None) or {}, "get", lambda _k: None)("code")
    if code == "model_not_supported":
        return True
    err = getattr(exc, "body", None)
    if isinstance(err, dict):
        e2 = err.get("error") or {}
        if isinstance(e2, dict) and e2.get("code") == "model_not_supported":
            return True
    return False


def _router_max_retries() -> int:
    raw = os.environ.get("LLM_JUDGE_ROUTER_MAX_RETRIES", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def _router_chat_completion(
    client: Any,
    model_id: str,
    user_msg: str,
    max_tokens: int,
) -> str:
    """Call HF Router with retries; raise RouterBillingError on 402 (no fallback helps)."""
    attempts = _router_max_retries()
    delay = float(os.environ.get("LLM_JUDGE_ROUTER_RETRY_DELAY", "3") or "3")
    last_err: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            comp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=max(64, min(512, max_tokens)),
                temperature=0,
            )
            return (comp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_err = exc
            if _is_router_billing_error(exc):
                raise RouterBillingError(
                    "HF Inference Provider credits depleted (HTTP 402). "
                    "Add prepaid credits: https://huggingface.co/settings/billing "
                    "or wait for your monthly included credits to reset. "
                    "Your batch checkpoint is saved — re-run the same cell after credits return."
                ) from exc
            if _is_router_transient_error(exc) and attempt + 1 < attempts:
                print(
                    f"[llm_judge] Transient API error (attempt {attempt + 1}/{attempts}): {exc}; "
                    f"retrying in {delay:.0f}s...",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            raise
    raise last_err  # type: ignore[misc]


def _score_row_router(
    user_msg: str,
    model_id: str,
    base_url: str,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    global _ROUTER_MODEL_CACHE
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) required for hf_router.")
    base = (base_url or "https://router.huggingface.co/v1").rstrip("/")
    client = OpenAI(api_key=token, base_url=base)

    if _ROUTER_MODEL_CACHE:
        candidates = [_ROUTER_MODEL_CACHE] + [
            m for m in _router_judge_model_candidates(model_id) if m != _ROUTER_MODEL_CACHE
        ]
    else:
        candidates = _router_judge_model_candidates(model_id)

    last_err: Optional[BaseException] = None
    for mid in candidates:
        try:
            raw = _router_chat_completion(client, mid, user_msg, max_tokens)
            if _ROUTER_MODEL_CACHE != mid:
                print(
                    f"[llm_judge] HF Router judge model: {mid!r} "
                    f"(set GP_JUDGE_MODEL or HF_ROUTER_MODEL_JUDGE to pin).",
                    flush=True,
                )
                _ROUTER_MODEL_CACHE = mid
            return _parse_judge_json(raw), raw
        except RouterBillingError:
            raise
        except Exception as exc:
            if _is_router_model_unsupported(exc):
                last_err = exc
                print(f"[llm_judge] Router model {mid!r} unavailable; trying next fallback.", flush=True)
                continue
            raise

    tried = ", ".join(repr(m) for m in candidates[:5])
    raise RuntimeError(
        f"No HF Router judge model available. Tried: {tried}. "
        "Enable Inference Providers at https://huggingface.co/settings/inference-providers "
        "or set GP_JUDGE_MODEL / HF_ROUTER_MODEL_JUDGE to a model your account supports "
        "(e.g. meta-llama/Llama-2-7b-chat-hf:featherless-ai)."
    ) from last_err


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _default_use_4bit() -> bool:
    env = os.environ.get("LLM_JUDGE_USE_4BIT", "").strip().lower()
    if env in ("0", "false", "no", "n"):
        return False
    if env in ("1", "true", "yes", "y"):
        return True
    # Kaggle T4: Qwen2.5-3B fits in fp16; bitsandbytes often not preinstalled.
    if os.path.isdir("/kaggle"):
        return False
    return True


def _load_local_judge(model_id: str, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        kwargs["token"] = token

    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    load_kw: Dict[str, Any] = {**kwargs}
    want_4bit = use_4bit and torch.cuda.is_available() and _bitsandbytes_available()
    if use_4bit and not want_4bit:
        print(
            "[llm_judge] Loading fp16 on GPU (bitsandbytes unavailable or disabled).",
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
        load_kw["device_map"] = "auto" if torch.cuda.is_available() else None

    def _load(**kw: Any):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except TypeError:
            if "dtype" in kw:
                kw = {**kw, "torch_dtype": kw.pop("dtype")}
            return AutoModelForCausalLM.from_pretrained(model_id, **kw)

    try:
        model = _load(**load_kw)
    except ImportError as exc:
        if "bitsandbytes" in str(exc).lower():
            print("[llm_judge] 4-bit load failed; retrying fp16 on GPU.", flush=True)
            load_kw = {**kwargs}
            load_kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            load_kw["device_map"] = "auto" if torch.cuda.is_available() else None
            load_kw.pop("quantization_config", None)
            model = _load(**load_kw)
        else:
            raise
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[llm_judge] Local judge loaded: {model_id!r}", flush=True)
    return model, tok


def _score_row_local(
    user_msg: str,
    model,
    tokenizer,
    max_new_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    import torch

    messages = [{"role": "user", "content": user_msg}]
    try:
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("no chat_template")
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted = user_msg

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max(32, min(256, max_new_tokens)),
            do_sample=False,
            pad_token_id=pad_id,
            use_cache=True,
        )
    new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_judge_json(raw), raw


def _row_key(row: Dict[str, Any]) -> str:
    """Stable id for one prediction row (includes run_folder for merged CSV exports)."""
    mname = str(row.get("model_name") or "")
    if not mname:
        mname = f"{row.get('model_key') or ''}_{'rag' if row.get('rag_flag') else 'norag'}"
    return "|".join(
        [
            str(row.get("run_folder") or ""),
            str(row.get("source_file") or ""),
            str(row.get("benchmark") or ""),
            str(row.get("question_id") or ""),
            mname,
        ]
    )


def _work_fingerprint(rows: List[Dict[str, Any]]) -> str:
    keys = sorted(_row_key(r) for r in rows)
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return digest[:16]


def _checkpoint_path(out_path: str) -> str:
    base, ext = os.path.splitext(out_path)
    if ext.lower() == ".json":
        return f"{base}.checkpoint.json"
    return f"{out_path}.checkpoint.json"


def _run_config_meta(
    *,
    predictions_path: str,
    source_format: str,
    judge_model: str,
    hf_router: bool,
    max_rows: int,
    seed: Optional[int],
    benchmark_filter: str,
    run_folder_filter: str,
    batch_size: int,
    batch_index: Optional[int],
    output_stem: str,
    work_rows: List[Dict[str, Any]],
    n_input_rows: int,
    row_offset: int = 0,
    row_limit: int = 0,
    person_tag: str = "",
    n_csv_total: int = 0,
) -> Dict[str, Any]:
    meta = {
        "predictions_path": predictions_path,
        "source_format": source_format,
        "judge_model": judge_model,
        "hf_router": hf_router,
        "max_rows": max_rows,
        "seed": seed,
        "benchmark_filter": benchmark_filter.strip(),
        "run_folder_filter": run_folder_filter.strip(),
        "batch_size": batch_size,
        "work_fingerprint": _work_fingerprint(work_rows),
        "n_work_rows": len(work_rows),
        "n_input_rows": n_input_rows,
        "row_offset": row_offset,
        "row_limit": row_limit,
        "person_tag": person_tag,
        "n_csv_total": n_csv_total,
    }
    if batch_size > 0:
        meta["output_stem"] = output_stem
        meta["batch_index"] = batch_index
    return meta


def _checkpoint_compatible(ckpt_meta: Dict[str, Any], run_meta: Dict[str, Any]) -> bool:
    keys = (
        "predictions_path",
        "source_format",
        "judge_model",
        "hf_router",
        "max_rows",
        "seed",
        "benchmark_filter",
        "run_folder_filter",
        "work_fingerprint",
    )
    if run_meta.get("batch_size", 0) > 0:
        keys = keys + (
            "batch_size",
            "output_stem",
            "batch_index",
            "row_offset",
            "row_limit",
            "person_tag",
            "n_csv_total",
        )
    for k in keys:
        if ckpt_meta.get(k) != run_meta.get(k):
            return False
    return True


def _load_checkpoint(
    ckpt_path: str,
    run_meta: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """Return judged rows by key and prior parse-failure count."""
    if not os.path.isfile(ckpt_path):
        return {}, 0
    with open(ckpt_path, encoding="utf-8") as f:
        payload = json.load(f)
    ckpt_meta = payload.get("meta") or {}
    if not _checkpoint_compatible(ckpt_meta, run_meta):
        print(
            "[llm_judge] Checkpoint options differ from this run; starting fresh.",
            flush=True,
        )
        return {}, 0
    by_key: Dict[str, Dict[str, Any]] = {}
    errors = 0
    for r in payload.get("rows") or []:
        if not isinstance(r, dict):
            continue
        by_key[_row_key(r)] = r
        if not isinstance(r.get("judge_scores"), dict):
            errors += 1
    print(
        f"[llm_judge] Resuming from {ckpt_path!r}: "
        f"{len(by_key)}/{run_meta.get('n_work_rows', '?')} rows already judged.",
        flush=True,
    )
    return by_key, errors


def _save_checkpoint(
    ckpt_path: str,
    *,
    run_meta: Dict[str, Any],
    by_key: Dict[str, Dict[str, Any]],
    work_rows: List[Dict[str, Any]],
    parse_failures: int,
) -> None:
    ordered = [by_key[_row_key(r)] for r in work_rows if _row_key(r) in by_key]
    meta = {
        **run_meta,
        "n_judged": len(ordered),
        "parse_failures": parse_failures,
        "checkpoint": True,
    }
    payload = {
        "meta": meta,
        "aggregate_by_model": _aggregate(ordered),
        "rows": ordered,
    }
    od = os.path.dirname(os.path.abspath(ckpt_path))
    if od:
        os.makedirs(od, exist_ok=True)
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ckpt_path)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        mname = str(r.get("model_name") or "unknown")
        scores = r.get("judge_scores")
        if not isinstance(scores, dict):
            continue
        by_model.setdefault(mname, []).append(scores)

    summary: Dict[str, Any] = {}
    for mname, lst in by_model.items():
        acc: Dict[str, List[float]] = {k: [] for k in _SCORE_KEYS}
        for d in lst:
            for k in _SCORE_KEYS:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    acc[k].append(float(v))
        summary[mname] = {
            k: (sum(acc[k]) / len(acc[k]) if acc[k] else None) for k in _SCORE_KEYS
        }
        for k in _SCORE_KEYS:
            vals = acc[k]
            if len(vals) >= 2:
                summary[mname][f"{k}_std"] = statistics.stdev(vals)
            elif len(vals) == 1:
                summary[mname][f"{k}_std"] = 0.0
            else:
                summary[mname][f"{k}_std"] = None
        summary[mname]["n_judged"] = len(lst)
    return summary


def _default_batch_size() -> int:
    raw = os.environ.get("LLM_JUDGE_BATCH_SIZE", "").strip()
    if raw.isdigit() and int(raw) >= 0:
        return int(raw)
    if os.path.isdir("/kaggle"):
        return _DEFAULT_BATCH_SIZE
    return 0


def _execute_judge_loop(
    *,
    work_rows: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
    ckpt_path: str,
    judge_model: str,
    hf_router: bool,
    hf_router_base_url: str,
    max_ctx_chars: int,
    use_4bit: bool,
    resume: bool,
    checkpoint_every: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """Judge pending rows; return (ordered rows, parse_errors, resumed)."""
    by_key: Dict[str, Dict[str, Any]] = {}
    errors = 0
    if resume:
        by_key, errors = _load_checkpoint(ckpt_path, run_meta)
    elif os.path.isfile(ckpt_path):
        print(f"[llm_judge] resume=False; ignoring existing {ckpt_path!r}", flush=True)

    pending: List[Dict[str, Any]] = []
    for row in work_rows:
        k = _row_key(row)
        if k in by_key:
            continue
        pending.append(row)
    n_total = len(work_rows)
    n_done = n_total - len(pending)
    if n_done:
        print(f"[llm_judge] Skipping {n_done} judged rows; {len(pending)} remaining.", flush=True)

    if not pending:
        out_rows = [by_key[_row_key(r)] for r in work_rows]
        return out_rows, errors, n_done > 0

    local_model = None
    local_tok = None
    if not hf_router:
        local_model, local_tok = _load_local_judge(judge_model, use_4bit=use_4bit)

    ckpt_every = max(1, int(checkpoint_every))
    for i, row in enumerate(pending):
        user_msg = _build_user_message(row, max_ctx_chars=max_ctx_chars)
        try:
            if hf_router:
                parsed, raw = _score_row_router(
                    user_msg, judge_model, hf_router_base_url, max_tokens=256
                )
            else:
                assert local_model is not None and local_tok is not None
                parsed, raw = _score_row_local(user_msg, local_model, local_tok, max_new_tokens=200)
        except RouterBillingError:
            _save_checkpoint(
                ckpt_path,
                run_meta=run_meta,
                by_key=by_key,
                work_rows=work_rows,
                parse_failures=errors,
            )
            done = len([r for r in work_rows if _row_key(r) in by_key])
            print(
                f"[llm_judge] STOPPED (HF credits): saved {done}/{n_total} rows in {ckpt_path!r}. "
                f"Re-run the same cell after adding credits — batch will resume at row {done + 1}.",
                flush=True,
            )
            raise

        rec = dict(row)
        rec["judge_model"] = judge_model
        rec["judge_raw"] = raw
        rec["judge_scores"] = parsed
        if parsed is None:
            errors += 1
        by_key[_row_key(row)] = rec
        global_idx = n_done + i + 1
        print(
            f"  [{global_idx}/{n_total}] {row.get('model_name')} parse_ok={parsed is not None}",
            flush=True,
        )
        if (i + 1) % ckpt_every == 0 or i == len(pending) - 1:
            _save_checkpoint(
                ckpt_path,
                run_meta=run_meta,
                by_key=by_key,
                work_rows=work_rows,
                parse_failures=errors,
            )

    out_rows = [by_key[_row_key(r)] for r in work_rows]
    return out_rows, errors, n_done > 0


def run_judge_batch(
    predictions_path: str,
    out_path: str,
    model_id: str,
    hf_router: bool,
    hf_router_base_url: str,
    batch_size: int,
    benchmark_filter: str,
    run_folder_filter: str,
    max_ctx_chars: int,
    use_4bit: bool,
    *,
    person: int = 0,
    row_offset: int = 0,
    row_limit: int = 0,
    slice_name: str = "",
    resume: bool = True,
    checkpoint_every: int = 1,
) -> Dict[str, Any]:
    slice_cfg: Optional[Dict[str, Any]] = None
    if _should_use_slice_preset(slice_name, person, row_offset, row_limit):
        slice_cfg = _resolve_active_slice(slice_name)
        predictions_path = _resolve_predictions_path(
            (predictions_path or str(slice_cfg.get("path") or "")).strip()
            or str(slice_cfg["path"]),
            person=0,
        )
        row_offset = int(slice_cfg["row_offset"])
        row_limit = int(slice_cfg["row_limit"])
        person = 0
        tag = str(slice_cfg.get("tag") or "")
        if tag:
            os.environ["LLM_JUDGE_OUTPUT_TAG"] = tag
    else:
        if person <= 0:
            person = _resolve_person_id(0)
        predictions_path = _resolve_predictions_path(predictions_path, person=person)
    payload = _load_predictions_payload(predictions_path)
    rows: List[Dict[str, Any]] = list(payload.get("rows") or [])
    source_format = str((payload.get("meta") or {}).get("source_format") or "")
    if not source_format:
        source_format = "csv" if _is_csv_path(predictions_path) else "json"

    if run_folder_filter.strip():
        rf = run_folder_filter.strip()
        rows = [r for r in rows if str(r.get("run_folder") or "").strip() == rf]
    if benchmark_filter.strip():
        bf = benchmark_filter.strip().lower()
        rows = [r for r in rows if str(r.get("benchmark") or "").lower() == bf]

    sorted_rows = _sort_rows_deterministic(rows)
    n_csv_total = len(sorted_rows)
    off, lim, person_tag, person_mode = _resolve_row_slice(
        n_csv_total, person=person, row_offset=row_offset, row_limit=row_limit
    )
    if slice_cfg and slice_cfg.get("tag"):
        person_tag = str(slice_cfg["tag"])
        person_mode = True
    work_rows = _apply_row_slice(sorted_rows, off, lim)
    n_input = len(work_rows)
    batches, batch_sizes = _split_batches(work_rows, batch_size)
    n_batches = len(batches)
    output_stem = _judge_output_stem(predictions_path, out_path, person_tag=person_tag)

    judge_model = _resolve_judge_model(model_id, hf_router=hf_router)
    if not judge_model:
        raise ValueError("Set model=... or GP_JUDGE_MODEL / HF_ROUTER_MODEL_JUDGE.")

    if hf_router:
        _try_load_hf_token()
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
            raise RuntimeError(
                "HF_TOKEN required for --hf_router on Kaggle. "
                "Add-ons → Secrets → enable HF_TOKEN, or set os.environ['HF_TOKEN'] in a cell above."
            )
    else:
        _try_load_hf_token()

    completed_rows = _count_completed_judge_rows(
        output_stem, n_batches, batch_sizes, person_mode=person_mode
    )
    search_dirs = _batch_search_dirs(output_stem, person_mode=person_mode)
    if completed_rows > 0 or person_mode:
        print(
            f"[llm_judge] Batch search dirs: {search_dirs[:5]}{'...' if len(search_dirs) > 5 else ''}",
            flush=True,
        )
    next_idx = _find_next_batch_index(
        output_stem, n_batches, batch_sizes, person_mode=person_mode
    )
    last_done = _last_complete_batch_index(
        output_stem, n_batches, batch_sizes, person_mode=person_mode
    )
    work_dir = os.path.dirname(os.path.abspath(output_stem))
    global_end = off + lim

    print(
        f"[llm_judge] Loaded {n_csv_total} rows ({source_format}) from {predictions_path!r}; "
        f"judging global rows {off}–{global_end - 1} ({n_input} rows)"
        f"{f' [{person_tag}]' if person_tag else ''}; "
        f"batch_size={batch_size} → {n_batches} batches; "
        f"{completed_rows}/{n_input} already on disk; "
        f"output_dir={work_dir!r}; "
        f"last_complete_batch={last_done if last_done is not None else 'none'}; "
        f"next_batch={next_idx if next_idx is not None else 'done'}; "
        f"judge_model={judge_model!r}.",
        flush=True,
    )

    if next_idx is None:
        manifest_path = f"{output_stem}_batch_manifest.json"
        manifest = {
            "predictions_path": predictions_path,
            "output_stem": output_stem,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "n_rows": n_input,
            "complete": True,
            "batch_files": [_batch_csv_path(output_stem, i) for i in range(n_batches)],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(
            f"[llm_judge] All {n_batches} batches complete ({n_input} rows). "
            f"Manifest: {manifest_path}",
            flush=True,
        )
        return manifest

    batch_rows = batches[next_idx]
    expected = batch_sizes[next_idx]
    batch_csv = _batch_csv_path(output_stem, next_idx)
    ckpt_path = _batch_checkpoint_path(output_stem, next_idx)

    print(
        f"[llm_judge] Processing batch {next_idx + 1}/{n_batches} "
        f"({len(batch_rows)} rows) → {batch_csv!r}",
        flush=True,
    )

    run_meta = _run_config_meta(
        predictions_path=predictions_path,
        source_format=source_format,
        judge_model=judge_model,
        hf_router=hf_router,
        max_rows=0,
        seed=None,
        benchmark_filter=benchmark_filter,
        run_folder_filter=run_folder_filter,
        batch_size=batch_size,
        batch_index=next_idx,
        output_stem=output_stem,
        work_rows=batch_rows,
        n_input_rows=n_input,
        row_offset=off,
        row_limit=lim,
        person_tag=person_tag,
        n_csv_total=n_csv_total,
    )

    out_rows, errors, resumed = _execute_judge_loop(
        work_rows=batch_rows,
        run_meta=run_meta,
        ckpt_path=ckpt_path,
        judge_model=judge_model,
        hf_router=hf_router,
        hf_router_base_url=hf_router_base_url,
        max_ctx_chars=max_ctx_chars,
        use_4bit=use_4bit,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )

    _write_judge_csv(batch_csv, out_rows)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)

    done_after = completed_rows + len(out_rows)
    remaining_batches = n_batches - (next_idx + 1)
    print(
        f"[llm_judge] Wrote {batch_csv!r} ({len(out_rows)} rows, parse_failures={errors}). "
        f"Progress: {done_after}/{n_input} rows; {remaining_batches} batch(es) left. Re-run for next batch.",
        flush=True,
    )

    return {
        "meta": {
            **run_meta,
            "batch_csv": batch_csv,
            "n_judged": len(out_rows),
            "parse_failures": errors,
            "resumed": resumed,
            "n_batches_total": n_batches,
            "n_batches_remaining": remaining_batches,
            "n_rows_done_total": done_after,
        },
        "aggregate_by_model": _aggregate(out_rows),
        "rows": out_rows,
    }


def run_judge(
    predictions_path: str,
    out_path: str,
    model_id: str,
    hf_router: bool,
    hf_router_base_url: str,
    max_rows: int,
    seed: Optional[int],
    benchmark_filter: str,
    run_folder_filter: str,
    max_ctx_chars: int,
    use_4bit: bool,
    *,
    batch_size: int = 0,
    person: int = 0,
    row_offset: int = 0,
    row_limit: int = 0,
    slice_name: str = "",
    resume: bool = True,
    checkpoint_every: int = 1,
) -> Dict[str, Any]:
    if batch_size > 0:
        return run_judge_batch(
            predictions_path=predictions_path,
            out_path=out_path,
            model_id=model_id,
            hf_router=hf_router,
            hf_router_base_url=hf_router_base_url,
            batch_size=batch_size,
            benchmark_filter=benchmark_filter,
            run_folder_filter=run_folder_filter,
            max_ctx_chars=max_ctx_chars,
            use_4bit=use_4bit,
            person=person,
            row_offset=row_offset,
            row_limit=row_limit,
            slice_name=slice_name,
            resume=resume,
            checkpoint_every=checkpoint_every,
        )

    predictions_path = _resolve_predictions_path(predictions_path, person=_resolve_person_id(0))
    payload = _load_predictions_payload(predictions_path)
    rows: List[Dict[str, Any]] = list(payload.get("rows") or [])
    source_format = str((payload.get("meta") or {}).get("source_format") or "")
    if not source_format:
        source_format = "csv" if _is_csv_path(predictions_path) else "json"

    if run_folder_filter.strip():
        rf = run_folder_filter.strip()
        rows = [r for r in rows if str(r.get("run_folder") or "").strip() == rf]
    if benchmark_filter.strip():
        bf = benchmark_filter.strip().lower()
        rows = [r for r in rows if str(r.get("benchmark") or "").lower() == bf]

    n_input = len(rows)
    if seed is not None:
        random.seed(seed)
    if max_rows > 0 and len(rows) > max_rows:
        rows = random.sample(rows, max_rows)

    work_rows = rows
    judge_model = _resolve_judge_model(model_id, hf_router=hf_router)
    if not judge_model:
        raise ValueError("Set model=... or GP_JUDGE_MODEL / HF_ROUTER_MODEL_JUDGE.")

    if hf_router:
        _try_load_hf_token()
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
            raise RuntimeError(
                "HF_TOKEN required for --hf_router on Kaggle. "
                "Add-ons → Secrets → enable HF_TOKEN, or set os.environ['HF_TOKEN'] in a cell above."
            )

    print(
        f"[llm_judge] Loaded {n_input} rows ({source_format}) from {predictions_path!r}; "
        f"judging {len(work_rows)} after filters/sample.",
        flush=True,
    )

    run_meta = _run_config_meta(
        predictions_path=predictions_path,
        source_format=source_format,
        judge_model=judge_model,
        hf_router=hf_router,
        max_rows=max_rows,
        seed=seed,
        benchmark_filter=benchmark_filter,
        run_folder_filter=run_folder_filter,
        batch_size=0,
        batch_index=None,
        output_stem="",
        work_rows=work_rows,
        n_input_rows=n_input,
    )
    ckpt_path = _checkpoint_path(out_path)
    out_rows, errors, resumed = _execute_judge_loop(
        work_rows=work_rows,
        run_meta=run_meta,
        ckpt_path=ckpt_path,
        judge_model=judge_model,
        hf_router=hf_router,
        hf_router_base_url=hf_router_base_url,
        max_ctx_chars=max_ctx_chars,
        use_4bit=use_4bit,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )

    meta = {
        **run_meta,
        "n_judged": len(out_rows),
        "parse_failures": errors,
        "resumed": resumed,
    }
    result = {"meta": meta, "aggregate_by_model": _aggregate(out_rows), "rows": out_rows}
    od = os.path.dirname(os.path.abspath(out_path))
    if od:
        os.makedirs(od, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    print(f"Wrote {out_path}", flush=True)
    return result


def _in_jupyter_or_ipython() -> bool:
    if any(m in sys.modules for m in ("IPython", "ipykernel")):
        return True
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def judge_predictions(
    predictions_json: str,
    *,
    out: str = "",
    model: str = "",
    hf_router: bool = False,
    hf_router_base_url: str = "",
    max_rows: int = 0,
    seed: Optional[int] = None,
    benchmark: str = "",
    run_folder: str = "",
    batch_size: int = -1,
    person: int = 0,
    row_offset: int = 0,
    row_limit: int = 0,
    slice: str = "",
    max_ctx_chars: int = 6000,
    use_4bit: Optional[bool] = None,
    resume: bool = True,
    checkpoint_every: int = 1,
) -> Dict[str, Any]:
    default_model = ""
    mid = _resolve_judge_model(model, hf_router=hf_router)
    slice_cfg: Optional[Dict[str, Any]] = None
    if _should_use_slice_preset(slice, person, row_offset, row_limit):
        slice_cfg = _resolve_active_slice(slice)
        pred_abs = _resolve_predictions_path(
            (predictions_json or str(slice_cfg.get("path") or "")).strip()
            or str(slice_cfg["path"]),
            person=0,
        )
        row_offset = int(slice_cfg["row_offset"])
        row_limit = int(slice_cfg["row_limit"])
        person = 0
        out_tag = str(slice_cfg.get("tag") or "")
    else:
        pid = person if person > 0 else _resolve_person_id(0)
        pred_abs = _resolve_predictions_path(predictions_json, person=pid)
        out_tag = _person_tag(pid) if pid > 0 else ""
    out_path = out.strip()
    if batch_size < 0:
        batch_size = _default_batch_size()
    if batch_size > 0:
        if not out_path:
            out_path = _judge_output_stem(pred_abs, "", person_tag=out_tag)
    elif not out_path:
        out_path = os.path.splitext(pred_abs)[0] + "_judge.json"
    else:
        out_path = os.path.abspath(out_path)
    if use_4bit is None:
        use_4bit = _default_use_4bit()
    return run_judge(
        predictions_path=pred_abs,
        out_path=out_path,
        model_id=mid,
        hf_router=hf_router,
        hf_router_base_url=(hf_router_base_url or os.environ.get("HF_ROUTER_BASE_URL", "")).strip(),
        max_rows=max_rows,
        seed=seed,
        benchmark_filter=benchmark,
        run_folder_filter=run_folder,
        max_ctx_chars=max_ctx_chars,
        use_4bit=use_4bit,
        batch_size=batch_size,
        person=person,
        row_offset=row_offset,
        row_limit=row_limit,
        slice_name=slice,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )


def _maybe_notebook_autorun() -> None:
    """
    Run judge after paste-in-cell load (default in Jupyter / IPython).

    - default → auto-find ``benchmark_results_all_predictions.json``
    - ``LLM_JUDGE_AUTO=/path/to/file.json`` → use that path (with fallback if missing)
    - ``LLM_JUDGE_AUTO=0`` or ``skip`` → no auto-run (call ``judge_predictions`` yourself)
    """
    raw = os.environ.get("LLM_JUDGE_AUTO", "").strip()
    if raw.lower() in ("0", "false", "no", "n", "skip"):
        return
    if raw.lower() in ("1", "true", "yes", "y", "auto") or not raw:
        path = ""
    else:
        path = raw
    max_rows = int(
        os.environ.get("LLM_JUDGE_MAX_ROWS", str(_DEFAULT_NOTEBOOK_MAX_ROWS))
        or str(_DEFAULT_NOTEBOOK_MAX_ROWS)
    )
    seed_raw = os.environ.get("LLM_JUDGE_SEED", "42").strip()
    seed: Optional[int] = (
        int(seed_raw) if seed_raw and (seed_raw.isdigit() or seed_raw.lstrip("-").isdigit()) else None
    )
    hf_router = _resolve_hf_router_flag(cli_hf_router=False)
    _log_judge_backend(hf_router)
    model = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    run_folder = os.environ.get("LLM_JUDGE_RUN_FOLDER", "").strip()
    no_resume = os.environ.get("LLM_JUDGE_NO_RESUME", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    ckpt_every = int(os.environ.get("LLM_JUDGE_CHECKPOINT_EVERY", "1") or "1")
    batch_size = _default_batch_size()
    if os.environ.get("LLM_JUDGE_BATCH_SIZE", "").strip().isdigit():
        batch_size = int(os.environ.get("LLM_JUDGE_BATCH_SIZE", "").strip())
    person = _default_person()
    slice_name = os.environ.get("LLM_JUDGE_SLICE", "").strip() or _default_slice_name()
    if _should_use_slice_preset(slice_name, person, 0, 0):
        slice_cfg = _resolve_active_slice(slice_name)
        resolved = _resolve_predictions_path(str(slice_cfg["path"]), person=0)
        slice_label = f" slice={slice_name!r} rows={slice_cfg['row_offset']}-{slice_cfg['row_offset'] + slice_cfg['row_limit'] - 1}"
    else:
        resolved = _resolve_predictions_path(path, person=person if person > 0 else _resolve_person_id(0))
        slice_label = ""
    print(
        f"[llm_judge] predictions={resolved!r} person={person}{slice_label} batch_size={batch_size} max_rows={max_rows} "
        f"hf_router={hf_router} run_folder={run_folder or 'all'} resume={not no_resume} "
        f"checkpoint_every={ckpt_every}",
        flush=True,
    )
    try:
        judge_predictions(
            resolved,
            model=model,
            hf_router=hf_router,
            max_rows=max_rows,
            seed=seed,
            run_folder=run_folder,
            batch_size=batch_size,
            person=person,
            slice=slice_name if slice_label else "",
            resume=not no_resume,
            checkpoint_every=ckpt_every,
        )
    except RouterBillingError as exc:
        print(f"[llm_judge] {exc}", flush=True)
        return


def _strip_jupyter_kernel_argv(argv: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "-f" and i + 1 < len(argv):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main() -> None:
    sys.argv = _strip_jupyter_kernel_argv(list(sys.argv))
    p = argparse.ArgumentParser(
        description="LLM-as-judge over *_predictions.json or benchmark_results_all_predictions_combined.csv."
    )
    p.add_argument(
        "predictions_path",
        nargs="?",
        default="",
        help="Path to predictions JSON or combined CSV (default: auto-find on Kaggle).",
    )
    p.add_argument("--out", default="", help="Output JSON (default: <predictions>_judge.json)")
    p.add_argument("--model", default="", help="HF model id. Default: GP_JUDGE_MODEL or router/local defaults.")
    p.add_argument("--hf_router", action="store_true", help="Use HF Inference Router (OpenAI API).")
    p.add_argument("--hf_router_base_url", default="", help="OpenAI API base URL.")
    p.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help=f"Rows per run/batch CSV (default: {_DEFAULT_BATCH_SIZE} on Kaggle, 0 = single JSON).",
    )
    p.add_argument("--max_rows", type=int, default=0, help="Random sample cap when batch_size=0 (0 = all).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed when max_rows is set.")
    p.add_argument("--benchmark", default="", help="Only rows with this benchmark name.")
    p.add_argument(
        "--run_folder",
        default="",
        help="Only rows with this run_folder (merged CSV: Bipro, Sifat, Medha, Salma).",
    )
    p.add_argument(
        "--slice",
        default="",
        help="Named row slice (e.g. hafijur30340 = rows 30340-31679; sifat28840 = rows 28840-31679).",
    )
    p.add_argument(
        "--person",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4),
        help="Team split: 1=rows 21000-26339, 2=26340-31679, etc. Ignored when --slice is set.",
    )
    p.add_argument(
        "--row_offset",
        type=int,
        default=0,
        help="Start row in sorted CSV (0-based). Overrides --person if both set with --row_limit.",
    )
    p.add_argument(
        "--row_limit",
        type=int,
        default=0,
        help="Max rows to judge from row_offset (0 = until end). Person 1 uses 5340.",
    )
    p.add_argument("--max_ctx_chars", type=int, default=6000, help="Max chars for context/RAG in prompt.")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit for local judge.")
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore checkpoint and re-judge all rows from scratch.",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Flush checkpoint JSON every N newly judged rows (default: 1).",
    )
    args, unknown = p.parse_known_args()
    if unknown:
        print("Ignoring extra argv:", unknown, flush=True)

    batch_size = args.batch_size if args.batch_size >= 0 else _default_batch_size()
    slice_name = (args.slice or os.environ.get("LLM_JUDGE_SLICE", "")).strip()
    if _should_use_slice_preset(slice_name, args.person, args.row_offset, args.row_limit):
        slice_cfg = _resolve_active_slice(slice_name)
        pred = _resolve_predictions_path(
            (args.predictions_path or str(slice_cfg["path"])).strip() or str(slice_cfg["path"]),
            person=0,
        )
        person = 0
        row_offset = int(slice_cfg["row_offset"])
        row_limit = int(slice_cfg["row_limit"])
        out_tag = str(slice_cfg.get("tag") or "")
    else:
        person = args.person if args.person > 0 else _default_person()
        pred = _resolve_predictions_path(args.predictions_path or "", person=person)
        row_offset = args.row_offset
        row_limit = args.row_limit
        out_tag = _person_tag(person) if person > 0 else ""
    out = args.out.strip()
    if batch_size > 0:
        if not out:
            out = _judge_output_stem(pred, "", person_tag=out_tag)
    elif not out:
        base = os.path.splitext(pred)[0]
        out = f"{base}_judge.json"
    default_model = ""
    hf_router = _resolve_hf_router_flag(cli_hf_router=args.hf_router)
    _log_judge_backend(hf_router)
    mid = _resolve_judge_model(args.model, hf_router=hf_router)
    try:
        run_judge(
            predictions_path=pred,
            out_path=out,
            model_id=mid,
            hf_router=hf_router,
            hf_router_base_url=(args.hf_router_base_url or os.environ.get("HF_ROUTER_BASE_URL", "")).strip(),
            max_rows=args.max_rows,
            seed=args.seed,
            benchmark_filter=args.benchmark,
            run_folder_filter=args.run_folder,
            max_ctx_chars=args.max_ctx_chars,
            use_4bit=False if args.no_4bit else _default_use_4bit(),
            batch_size=batch_size,
            person=person,
            row_offset=row_offset,
            row_limit=row_limit,
            slice_name=slice_name,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
        )
    except RouterBillingError as exc:
        print(f"[llm_judge] {exc}", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    if _in_jupyter_or_ipython():
        _maybe_notebook_autorun()
        if os.environ.get("LLM_JUDGE_AUTO", "").strip().lower() in (
            "0",
            "false",
            "no",
            "n",
            "skip",
        ):
            print(
                "[llm_judge] Person 3 (rows 31,680–37,019, fatinshadab CSV): re-run cell for each batch.\n"
                "  judge_predictions('', person=3, batch_size=500)",
                flush=True,
            )
    else:
        main()
