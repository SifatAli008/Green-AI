#!/usr/bin/env python3
"""
Production benchmark orchestrator for NVIDIA L4 (Vast.ai / single-GPU).

Runs 8 models × 2 configs (NoRAG / RAG) × 3 datasets = 48 jobs with:
  - Per-query latency_seconds in predictions CSV
  - NVML GPU energy (pynvml) per query
  - Unsloth 4-bit loading when available (bitsandbytes fallback via real_model_runner)
  - Paper RAG corpus: chunks_paper_556.jsonl + index_paper_556.faiss, RAG_TOP_K=3

Usage (on Vast.ai instance after cloning repo + setting HF_TOKEN):
  cd GreenPaper_Kaggle_Benchmarks
  pip install -r requirements_vast_l4.txt   # or let --install_deps handle it
  python run_vast_l4_benchmarks.py --output_dir ./vast_results --seed 42 --max_items 1000

Resume: re-run the same command; completed CSVs are skipped unless --force.

Smoke test (cheap preflight before full 1000-q campaign):
  python run_vast_l4_benchmarks.py --smoke_test

Skip MedGemma if HF approval pending:
  python run_vast_l4_benchmarks.py --exclude_models MedGemma4B
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & imports from existing codebase
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("GP_BENCH_TQDM", "1")
os.environ.setdefault("RAG_REPAIR_PROMPT", "1")
os.environ.setdefault("RAG_TOP_K", "3")
os.environ.setdefault("RAG_CONTEXT_MAX_CHARS", "4500")

LOGGER = logging.getLogger("vast_l4")

MCQ_BENCHMARKS = frozenset({"medqa", "mmlu_med", "medmcqa", "custom_mcq"})

MEDICAL_SLMS: List[Dict[str, str]] = [
    {"id": "microsoft/MediPhi-Instruct", "short": "MediPhi", "family": "slm"},
    {"id": "google/medgemma-4b-it", "short": "MedGemma4B", "family": "slm"},
    {"id": "dmis-lab/meerkat-7b-v1.0", "short": "Meerkat7B", "family": "slm"},
    {"id": "BioMistral/BioMistral-7B-DARE", "short": "BioMistral7B", "family": "slm"},
    {"id": "internistai/base-7b-v0.2", "short": "Internist7B", "family": "slm"},
]

GENERAL_LLMS: List[Dict[str, str]] = [
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "short": "Mistral7B", "family": "llm"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "short": "Llama31_8B", "family": "llm"},
    {"id": "google/gemma-2-9b-it", "short": "Gemma2_9B", "family": "llm"},
]

ALL_MODELS = MEDICAL_SLMS + GENERAL_LLMS

# Hub models that require HF_TOKEN + access approval (401 without it).
GATED_MODEL_MARKERS: Tuple[str, ...] = (
    "meta-llama/",
    "google/gemma",
    "google/medgemma",
)

DATASETS: List[Dict[str, str]] = [
    {"name": "medqa", "label": "MedQA-USMLE"},
    {"name": "pubmedqa", "label": "PubMedQA"},
    {"name": "mmlu_med", "label": "MMLU-Medical"},
]

CONFIGS: List[Dict[str, Any]] = [
    {"name": "NoRAG", "use_rag": False},
    {"name": "RAG", "use_rag": True},
]

VAST_L4_DEPS = (
    "torch",
    "transformers>=4.43.0,<5",
    "accelerate>=0.26.0",
    "bitsandbytes>=0.46.1",
    "sentence-transformers>=2.2.2",
    "faiss-cpu",
    "datasets",
    "huggingface-hub>=0.23.0",
    "safetensors>=0.4.0",
    "pynvml",
    "tqdm",
    "numpy",
    "scipy",
)


@dataclass
class JobSpec:
    model: Dict[str, str]
    dataset: str
    dataset_label: str
    config_name: str
    use_rag: bool
    output_prefix: str

    @property
    def job_id(self) -> str:
        return f"{self.model['short']}_{self.dataset}_{self.config_name}"


# ---------------------------------------------------------------------------
# NVML energy metering
# ---------------------------------------------------------------------------
class NvmlEnergyMeter:
    """
    Per-query GPU energy via NVML Total Energy Consumption.

    API note (pynvml): ``nvmlDeviceGetTotalEnergyConsumption`` returns cumulative
    energy in **millijoules** since driver load (resets on power cycle / driver reload).
    Delta between start/stop is converted to joules (÷1000) for CSV columns.
    Falls back to mean power × elapsed when the counter is unsupported or wraps.
    """

    def __init__(self, device_index: int = 0) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self.device_index = device_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        raw_name = pynvml.nvmlDeviceGetName(self.handle)
        self.gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        self._supports_total_energy = self._probe_total_energy()
        self._e0_mj: Optional[int] = None
        self._t0: float = 0.0
        self._power_samples: List[float] = []
        if self._supports_total_energy:
            baseline = self._read_energy_mj()
            LOGGER.info(
                "NVML energy counter: supported | baseline=%s mJ | GPU=%s",
                baseline,
                self.gpu_name,
            )
        else:
            LOGGER.warning(
                "NVML GetTotalEnergyConsumption not supported on %s; using power×time fallback.",
                self.gpu_name,
            )

    def _probe_total_energy(self) -> bool:
        try:
            self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return True
        except self._pynvml.NVMLError:
            return False

    def _sync_cuda(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _read_energy_mj(self) -> Optional[int]:
        """Cumulative GPU energy in millijoules (NVML API)."""
        if not self._supports_total_energy:
            return None
        try:
            return int(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except self._pynvml.NVMLError:
            return None

    def _read_power_w(self) -> Optional[float]:
        try:
            mw = self._pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return float(mw) / 1000.0
        except self._pynvml.NVMLError:
            return None

    def read_baseline_mj(self) -> Optional[int]:
        """Current cumulative energy counter (for logging after reset)."""
        return self._read_energy_mj()

    def start(self) -> None:
        self._sync_cuda()
        self._e0_mj = self._read_energy_mj()
        self._t0 = time.perf_counter()
        self._power_samples = []
        pw = self._read_power_w()
        if pw is not None:
            self._power_samples.append(pw)

    def stop(self) -> Dict[str, Any]:
        self._sync_cuda()
        pw = self._read_power_w()
        if pw is not None:
            self._power_samples.append(pw)
        elapsed = max(time.perf_counter() - self._t0, 1e-9)

        energy_j: Optional[float] = None
        counter_wrap = False
        if self._supports_total_energy and self._e0_mj is not None:
            e1 = self._read_energy_mj()
            if e1 is not None:
                if e1 >= self._e0_mj:
                    delta_mj = e1 - self._e0_mj
                    energy_j = delta_mj / 1000.0
                else:
                    counter_wrap = True
                    LOGGER.debug(
                        "NVML energy counter decreased (%s -> %s mJ); counter may have reset.",
                        self._e0_mj,
                        e1,
                    )
        if energy_j is None and self._power_samples:
            mean_w = sum(self._power_samples) / len(self._power_samples)
            energy_j = mean_w * elapsed
        elif energy_j is None:
            energy_j = 0.0

        kwh = energy_j / 3_600_000.0
        out: Dict[str, Any] = {
            "energy_joules_nvml": round(energy_j, 6),
            "gpu_energy_kwh_nvml": round(kwh, 12),
            "gpu_name": self.gpu_name,
            "nvml_energy_counter_wrap": counter_wrap,
        }
        if self._power_samples:
            out["gpu_power_watts_mean_nvml"] = round(
                sum(self._power_samples) / len(self._power_samples), 2
            )
        return out

    def shutdown(self) -> None:
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reproducibility & environment
# ---------------------------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def reset_gpu_energy_counters(device_index: int = 0) -> bool:
    """
    Reset NVML cumulative energy counter via nvidia-smi.

    Recommended at the start of each model block on Vast.ai when per-query
    deltas look inconsistent (counter carries over from prior workloads).
    """
    import subprocess

    for cmd in (
        ["nvidia-smi", "-i", str(device_index), "-e", "0"],
        ["nvidia-smi", "-e", "0"],
    ):
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                LOGGER.info("Reset GPU energy counter: %s", " ".join(cmd))
                return True
            LOGGER.debug(
                "Energy reset attempt failed (%s): %s",
                " ".join(cmd),
                (result.stderr or result.stdout or "").strip(),
            )
        except FileNotFoundError:
            LOGGER.warning("nvidia-smi not found; cannot reset energy counters.")
            return False
        except subprocess.TimeoutExpired:
            LOGGER.warning("nvidia-smi energy reset timed out for %s", cmd)
    LOGGER.warning(
        "Could not reset GPU energy counters (non-fatal). "
        "Per-query NVML deltas may include prior workload if counter was not reset."
    )
    return False


def get_hf_token() -> str:
    return (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    )


def model_requires_hf_gate(model_id: str) -> bool:
    low = model_id.lower()
    return any(marker in low for marker in GATED_MODEL_MARKERS)


def require_hf_token_for_models(models: List[Dict[str, str]], *, strict: bool = True) -> None:
    """Fail fast if gated checkpoints are in the run list but HF_TOKEN is missing."""
    gated = [m for m in models if model_requires_hf_gate(m["id"])]
    token = get_hf_token()
    if gated and not token:
        ids = ", ".join(m["id"] for m in gated)
        raise ValueError(
            "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set but required for gated models: "
            f"{ids}. Export your token before launching, e.g. export HF_TOKEN=hf_..."
        )
    if not strict or not token:
        return
    try:
        from huggingface_hub import HfApi

        who = HfApi(token=token).whoami()
        LOGGER.info("HF hub auth OK (user=%s)", who.get("name", who.get("email", "?")))
    except Exception as exc:
        raise ValueError(
            f"HF_TOKEN is set but Hugging Face authentication failed: {exc}"
        ) from exc


def verify_hf_model_access(models: List[Dict[str, str]]) -> None:
    """Warn when gated models (e.g. MedGemma) may lack Hub approval."""
    token = get_hf_token()
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        for m in models:
            mid = m["id"]
            if not model_requires_hf_gate(mid):
                continue
            try:
                api.model_info(mid, token=token)
                LOGGER.info("HF access OK for %s", mid)
            except Exception as exc:
                err = str(exc).lower()
                if "401" in err or "403" in err or "gated" in err or "authorized" in err:
                    LOGGER.error(
                        "HF access NOT granted for %s (%s). "
                        "Skip with --exclude_models %s or replace for first batch.",
                        mid,
                        exc,
                        m["short"],
                    )
                else:
                    LOGGER.warning("Could not verify HF access for %s: %s", mid, exc)
    except ImportError:
        LOGGER.debug("huggingface_hub not available for model access preflight.")


def verify_l4_gpu(*, meter: Optional[NvmlEnergyMeter] = None, strict: bool = True) -> str:
    """Validate CUDA + NVML report an NVIDIA L4 (strict by default)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script requires a GPU instance.")

    torch_name = torch.cuda.get_device_name(0)
    nvml_name = meter.gpu_name if meter is not None else ""
    gpu_name = nvml_name or torch_name

    if strict and "l4" not in gpu_name.lower() and "l4" not in torch_name.lower():
        raise AssertionError(
            f"Expected NVIDIA L4 GPU, got NVML={nvml_name!r} torch={torch_name!r}. "
            "Use --allow_non_l4 only for local debugging."
        )

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    LOGGER.info(
        "GPU verified: %s | torch=%s | VRAM: %.1f GB | CC %d.%d",
        gpu_name,
        torch_name,
        vram_gb,
        props.major,
        props.minor,
    )
    return gpu_name


def install_deps_if_requested(do_install: bool) -> None:
    if not do_install:
        return
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "-U", *VAST_L4_DEPS]
    LOGGER.info("Installing dependencies: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    # Unsloth is optional; install separately if desired
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "unsloth", "xformers", "--no-deps"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        LOGGER.info("Optional: unsloth installed.")
    except Exception:
        LOGGER.info("Optional: unsloth not installed (will use bitsandbytes fallback).")


def slugify_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))


def verify_rag_corpus(rag_index_dir: Path) -> None:
    """Fail fast if paper RAG sidecars are missing."""
    from eval_benchmarks import _paper_rag_file_pair, _repo_kaggle_working_rag_index

    roots = [rag_index_dir]
    repo = _repo_kaggle_working_rag_index()
    if repo:
        roots.append(Path(repo))
    for root in roots:
        pair = _paper_rag_file_pair(str(root))
        if pair:
            LOGGER.info("RAG corpus found under %s", root)
            return
    raise FileNotFoundError(
        "Paper RAG corpus missing. Expected chunks_paper_556.jsonl + index_paper_556.faiss under "
        f"{rag_index_dir} or kaggle_working/rag_index/"
    )


def pin_paper_rag_corpus(rag_index_dir: Path) -> None:
    """Pin env to paper 556-chunk index (same logic as eval_benchmarks --rag_paper_corpus)."""
    from eval_benchmarks import _paper_rag_file_pair, _pin_rag_files, _repo_kaggle_working_rag_index

    roots: List[str] = []
    if rag_index_dir.is_dir():
        roots.append(str(rag_index_dir.resolve()))
    repo_rag = _repo_kaggle_working_rag_index()
    if repo_rag:
        roots.append(repo_rag)
    seen: set[str] = set()
    for d in roots:
        ad = os.path.abspath(d)
        if ad in seen:
            continue
        seen.add(ad)
        pair = _paper_rag_file_pair(ad)
        if pair:
            ip, cp = pair
            _pin_rag_files(ad, ip, cp)
            os.environ["GP_RAG_PAPER_556"] = "1"
            os.environ["RAG_TOP_K"] = "3"
            LOGGER.info("RAG paper corpus pinned:\n  chunks: %s\n  index:  %s", cp, ip)
            return
    raise FileNotFoundError(
        "Paper RAG corpus not found. Place chunks_paper_556.jsonl and index_paper_556.faiss "
        f"under {rag_index_dir} or kaggle_working/rag_index/"
    )


def clear_rag_pins() -> None:
    for key in (
        "RAG_INDEX_DIR",
        "RAG_FAISS_INDEX",
        "RAG_CHUNKS_JSONL",
        "GP_RAG_PAPER_556",
    ):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Model loading (Unsloth → bitsandbytes fallback)
# ---------------------------------------------------------------------------
def load_model_for_benchmark(
    model_id: str,
    *,
    use_unsloth: bool = True,
    use_4bit: bool = True,
) -> Tuple[Any, Any, str]:
    """
    Load one model for inference. Returns (model, tokenizer, backend_label).
    Sets GP_MODEL_SLM so real_model_runner.load_one_model can be used as fallback.
    """
    os.environ["GP_MODEL_SLM"] = model_id
    os.environ["GP_MODEL_LLM"] = model_id

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            LOGGER.info("Loading %s via Unsloth 4-bit...", model_id)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=use_4bit,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
            FastLanguageModel.for_inference(model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer, "unsloth_4bit"
        except Exception as exc:
            LOGGER.warning("Unsloth load failed for %s: %s — falling back to bitsandbytes.", model_id, exc)

    from real_model_runner import load_one_model

    LOGGER.info("Loading %s via real_model_runner (bitsandbytes 4-bit)...", model_id)
    model, tokenizer = load_one_model("slm", use_4bit=use_4bit)
    backend = "bitsandbytes_4bit" if use_4bit else "fp16"
    return model, tokenizer, backend


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def prewarm_rag_index() -> None:
    from real_model_runner import ensure_rag_index_env, prewarm_faiss_index, print_startup_diagnostics

    ensure_rag_index_env()
    print_startup_diagnostics()
    if not prewarm_faiss_index():
        LOGGER.warning("FAISS prewarm failed — check faiss-cpu and RAG paths.")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------
def build_jobs(
    models: List[Dict[str, str]],
    datasets: List[Dict[str, str]],
    configs: List[Dict[str, Any]],
    output_dir: Path,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for model in models:
        for ds in datasets:
            for cfg in configs:
                prefix = output_dir / f"{model['short']}_{ds['name']}_{cfg['name']}"
                jobs.append(
                    JobSpec(
                        model=model,
                        dataset=ds["name"],
                        dataset_label=ds["label"],
                        config_name=cfg["name"],
                        use_rag=bool(cfg["use_rag"]),
                        output_prefix=str(prefix),
                    )
                )
    return jobs


def make_run_fn(
    models_dict: Dict[str, Tuple[Any, Any]],
    meter: NvmlEnergyMeter,
    quantization_backend: str,
) -> Callable[[str, str, bool], Dict[str, Any]]:
    from real_model_runner import run_single

    def run_fn(model_key: str, question: str, use_rag: bool) -> Dict[str, Any]:
        meter.start()
        out = run_single(question, model_key=model_key, use_rag=use_rag, models_dict=models_dict)
        telem = meter.stop()
        if out.get("latency_seconds") in (None, ""):
            out["latency_seconds"] = out.get("latency")
        out.update(telem)
        out["quantization_backend"] = quantization_backend
        return out

    return run_fn


def run_single_job(
    job: JobSpec,
    *,
    max_items: int,
    seed: int,
    rag_index_dir: Path,
    meter: NvmlEnergyMeter,
    models_dict: Dict[str, Tuple[Any, Any]],
    quantization_backend: str,
) -> Dict[str, Any]:
    from eval_benchmarks import (
        PUBMEDQA_LABELED_SPLIT,
        _evaluate_free_text_for_model,
        _evaluate_mcq_for_model,
        _load_items,
        _model_ids_snapshot,
        _set_bootstrap_rng,
        _write_prediction_artifacts,
    )

    t_job = time.perf_counter()
    LOGGER.info("=" * 72)
    LOGGER.info(
        "JOB %s | model=%s | dataset=%s | config=%s",
        job.job_id,
        job.model["id"],
        job.dataset,
        job.config_name,
    )

    if job.use_rag:
        pin_paper_rag_corpus(rag_index_dir)
        prewarm_rag_index()
    else:
        clear_rag_pins()

    _set_bootstrap_rng(seed)
    items = _load_items(job.dataset, "", max_items, seed)
    LOGGER.info("Loaded n=%d items (max_items=%d, seed=%d)", len(items), max_items, seed)

    os.environ["GP_MODEL_SLM"] = job.model["id"]
    model_ids = {
        "slm_model_id": job.model["id"],
        "llm_model_id": job.model["id"],
    }

    cfg_tuple = (f"{job.model['short']}_{job.config_name}", "slm", job.use_rag)
    run_fn = make_run_fn(models_dict, meter, quantization_backend)

    if job.dataset in MCQ_BENCHMARKS:
        metrics, rows = _evaluate_mcq_for_model(
            items,
            run_fn,
            job.dataset,
            "slm",
            model_ids=model_ids,
            active_configs=[cfg_tuple],
        )
    else:
        metrics, rows = _evaluate_free_text_for_model(
            items,
            run_fn,
            job.dataset,
            "slm",
            model_ids=model_ids,
            active_configs=[cfg_tuple],
        )

    cfg_key = cfg_tuple[0]
    run_metrics = metrics.get(cfg_key, {})
    acc_mean = run_metrics.get("mean")
    if acc_mean is None:
        acc_mean = run_metrics.get("pubmedqa_label_accuracy_mean")
    meta = {
        "job_id": job.job_id,
        "model_id": job.model["id"],
        "model_family": job.model["family"],
        "dataset": job.dataset,
        "dataset_label": job.dataset_label,
        "config": job.config_name,
        "use_rag": job.use_rag,
        "max_items": max_items,
        "subset_seed": seed,
        "n_items": len(items),
        "quantization_backend": quantization_backend,
        "gpu_name": meter.gpu_name,
        "rag_top_k": int(os.environ.get("RAG_TOP_K", "3")),
        "pubmedqa_split": PUBMEDQA_LABELED_SPLIT if job.dataset == "pubmedqa" else "",
        "metrics": run_metrics,
        "wall_seconds": round(time.perf_counter() - t_job, 2),
    }
    _write_prediction_artifacts(job.output_prefix, rows, meta)

    # Summary stats for this job
    latencies = [float(r["latency_seconds"]) for r in rows if r.get("latency_seconds") not in (None, "")]
    energies = [float(r["energy_joules_nvml"]) for r in rows if r.get("energy_joules_nvml") not in (None, "")]
    correct = [r for r in rows if r.get("label_correct") in (True, 1, 1.0)]
    summary = {
        "job_id": job.job_id,
        "n_rows": len(rows),
        "accuracy": acc_mean,
        "accuracy_ci": (
            run_metrics.get("ci_lower") or run_metrics.get("pubmedqa_label_accuracy_ci_lower"),
            run_metrics.get("ci_upper") or run_metrics.get("pubmedqa_label_accuracy_ci_upper"),
        ),
        "label_accuracy": run_metrics.get("pubmedqa_label_accuracy_mean"),
        "mean_latency_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "total_energy_kwh_nvml": round(sum(energies) / 3_600_000.0, 8) if energies else None,
        "mean_energy_j_nvml": round(sum(energies) / len(energies), 6) if energies else None,
        "output_csv": f"{job.output_prefix}_predictions.csv",
        "wall_seconds": meta["wall_seconds"],
        "status": "ok",
    }
    return summary


def print_final_summary(summaries: List[Dict[str, Any]]) -> None:
    LOGGER.info("\n" + "=" * 72)
    LOGGER.info("BENCHMARK CAMPAIGN SUMMARY (%d jobs)", len(summaries))
    LOGGER.info("=" * 72)
    ok = [s for s in summaries if s.get("status") == "ok"]
    skipped = [s for s in summaries if s.get("status") == "skipped"]
    failed = [s for s in summaries if s.get("status") == "failed"]

    hdr = f"{'Job':<40} {'Acc':>7} {'Lat(s)':>8} {'E/query(J)':>12} {'N':>5}"
    LOGGER.info(hdr)
    LOGGER.info("-" * len(hdr))
    for s in ok:
        acc = s.get("accuracy")
        if acc is None:
            acc = s.get("label_accuracy")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) and acc == acc else "n/a"
        lat = s.get("mean_latency_s")
        lat_s = f"{lat:.3f}" if lat is not None else "n/a"
        ej = s.get("mean_energy_j_nvml")
        ej_s = f"{ej:.4f}" if ej is not None else "n/a"
        LOGGER.info(
            "%-40s %7s %8s %12s %5d",
            s["job_id"][:40],
            acc_s,
            lat_s,
            ej_s,
            s.get("n_rows", 0),
        )

    LOGGER.info(
        "\nCompleted: %d | Skipped: %d | Failed: %d",
        len(ok),
        len(skipped),
        len(failed),
    )
    if failed:
        LOGGER.info("Failures:")
        for s in failed:
            LOGGER.info("  %s: %s", s["job_id"], s.get("error", "unknown"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vast.ai L4 benchmark campaign: 8 models × 2 configs × 3 datasets."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "vast_l4_results"),
        help="Directory for per-job predictions CSV/JSON.",
    )
    p.add_argument("--seed", type=int, default=42, help="Fixed subset seed (reproducible 1000-q draws).")
    p.add_argument("--max_items", type=int, default=1000, help="Questions per dataset (0 = full split).")
    p.add_argument(
        "--rag_index_dir",
        type=str,
        default=str(SCRIPT_DIR / "kaggle_working" / "rag_index"),
        help="Folder containing chunks_paper_556.jsonl + index_paper_556.faiss.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model short names (e.g. MediPhi,MedGemma4B) or 'all'.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated: medqa,pubmedqa,mmlu_med or 'all'.",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="all",
        help="NoRAG,RAG or 'all'.",
    )
    p.add_argument("--no_unsloth", action="store_true", help="Skip Unsloth; use bitsandbytes only.")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit (may OOM on L4 for 7B+).")
    p.add_argument("--force", action="store_true", help="Re-run jobs even if output CSV exists.")
    p.add_argument("--install_deps", action="store_true", help="pip install required packages before run.")
    p.add_argument("--allow_non_l4", action="store_true", help="Skip L4 GPU assertion (debug only).")
    p.add_argument(
        "--skip_hf_token_check",
        action="store_true",
        help="Do not require HF_TOKEN (not recommended; gated models will fail at load).",
    )
    p.add_argument(
        "--exclude_models",
        type=str,
        default="",
        help="Comma-separated short names to skip (e.g. MedGemma4B if HF approval pending).",
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Preflight: force --max_items 10 and default output_dir .../smoke_test (48 jobs × 10 q).",
    )
    p.add_argument(
        "--no_energy_reset",
        action="store_true",
        help="Skip nvidia-smi energy counter reset at each model load.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print job list and exit.")
    p.add_argument("--log_file", type=str, default="", help="Optional log file path.")
    return p.parse_args()


def filter_models(spec: str, exclude: str = "") -> List[Dict[str, str]]:
    if spec.strip().lower() == "all":
        out = list(ALL_MODELS)
    else:
        wanted = {x.strip() for x in spec.split(",") if x.strip()}
        out = [m for m in ALL_MODELS if m["short"] in wanted or m["id"] in wanted]
        if len(out) != len(wanted):
            missing = wanted - {m["short"] for m in out} - {m["id"] for m in out}
            raise ValueError(f"Unknown model(s): {missing}")
    if exclude.strip():
        excl = {x.strip() for x in exclude.split(",") if x.strip()}
        out = [m for m in out if m["short"] not in excl and m["id"] not in excl]
        if excl:
            LOGGER.info("Excluded models: %s", ", ".join(sorted(excl)))
    if not out:
        raise ValueError("No models left after --models / --exclude_models filters.")
    return out


def filter_datasets(spec: str) -> List[Dict[str, str]]:
    if spec.strip().lower() == "all":
        return list(DATASETS)
    wanted = {x.strip() for x in spec.split(",") if x.strip()}
    out = [d for d in DATASETS if d["name"] in wanted]
    if len(out) != len(wanted):
        raise ValueError(f"Unknown dataset(s): {wanted - {d['name'] for d in out}}")
    return out


def filter_configs(spec: str) -> List[Dict[str, Any]]:
    if spec.strip().lower() == "all":
        return list(CONFIGS)
    wanted = {x.strip() for x in spec.split(",") if x.strip()}
    out = [c for c in CONFIGS if c["name"] in wanted]
    if len(out) != len(wanted):
        raise ValueError(f"Unknown config(s): {wanted - {c['name'] for c in out}}")
    return out


def setup_logging(log_file: str) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    default_output = str(SCRIPT_DIR / "vast_l4_results")
    if args.smoke_test:
        args.max_items = 10
        if args.output_dir == default_output:
            args.output_dir = str(SCRIPT_DIR / "vast_l4_results" / "smoke_test")
        LOGGER.info("SMOKE TEST mode: max_items=10, output_dir=%s", args.output_dir)

    install_deps_if_requested(args.install_deps)
    set_global_seeds(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_index_dir = Path(args.rag_index_dir).resolve()

    models = filter_models(args.models, exclude=args.exclude_models)
    datasets = filter_datasets(args.datasets)
    configs = filter_configs(args.configs)
    jobs = build_jobs(models, datasets, configs, output_dir)

    if args.dry_run:
        LOGGER.info(
            "DRY RUN | %d models × %d datasets × %d configs = %d jobs | seed=%d max_items=%d",
            len(models),
            len(datasets),
            len(configs),
            len(jobs),
            args.seed,
            args.max_items,
        )
        if not args.skip_hf_token_check:
            require_hf_token_for_models(models, strict=False)
        for j in jobs:
            LOGGER.info("  [dry-run] %s -> %s_predictions.csv", j.job_id, j.output_prefix)
        LOGGER.info(
            "MMLU-Med note: dev split may yield <1000 items; document actual n in paper methods."
        )
        return 0

    if not args.skip_hf_token_check:
        require_hf_token_for_models(models, strict=True)
        verify_hf_model_access(models)

    verify_rag_corpus(rag_index_dir)

    meter = NvmlEnergyMeter(device_index=0)
    verify_l4_gpu(meter=meter, strict=not args.allow_non_l4)

    LOGGER.info(
        "Campaign: %d models × %d datasets × %d configs = %d jobs",
        len(models),
        len(datasets),
        len(configs),
        len(jobs),
    )
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Seed=%d max_items=%d RAG_TOP_K=%s", args.seed, args.max_items, os.environ.get("RAG_TOP_K"))
    if args.max_items > 0:
        LOGGER.info(
            "Dataset size note: MMLU-Med dev may contribute fewer than max_items; "
            "actual n is logged per job."
        )

    summaries: List[Dict[str, Any]] = []
    manifest_path = output_dir / "campaign_manifest.jsonl"

    try:
        # Group jobs by model so each checkpoint loads once (6 jobs × 1000q, not 48 loads).
        jobs_by_model: Dict[str, List[JobSpec]] = {}
        for job in jobs:
            jobs_by_model.setdefault(job.model["id"], []).append(job)

        job_index = 0
        for model_id, model_jobs in jobs_by_model.items():
            LOGGER.info("=" * 72)
            LOGGER.info("Loading model once for %d jobs: %s", len(model_jobs), model_id)

            if not args.no_energy_reset:
                reset_gpu_energy_counters(meter.device_index)
                baseline_mj = meter.read_baseline_mj()
                if baseline_mj is not None:
                    LOGGER.info("NVML energy baseline after reset: %s mJ", baseline_mj)

            pending = []
            for job in model_jobs:
                job_index += 1
                csv_path = Path(f"{job.output_prefix}_predictions.csv")
                if csv_path.is_file() and not args.force:
                    LOGGER.info("[%d/%d] SKIP %s (exists)", job_index, len(jobs), job.job_id)
                    summaries.append(
                        {"job_id": job.job_id, "status": "skipped", "output_csv": str(csv_path)}
                    )
                else:
                    pending.append((job_index, job))

            if not pending:
                continue

            model, tokenizer, q_backend = load_model_for_benchmark(
                model_id,
                use_unsloth=not args.no_unsloth,
                use_4bit=not args.no_4bit,
            )
            models_dict = {"slm": (model, tokenizer)}

            try:
                for job_index, job in pending:
                    LOGGER.info("[%d/%d] START %s", job_index, len(jobs), job.job_id)
                    try:
                        summary = run_single_job(
                            job,
                            max_items=args.max_items,
                            seed=args.seed,
                            rag_index_dir=rag_index_dir,
                            meter=meter,
                            models_dict=models_dict,
                            quantization_backend=q_backend,
                        )
                        summaries.append(summary)
                        with open(manifest_path, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps(summary, ensure_ascii=False) + "\n")
                        LOGGER.info(
                            "[%d/%d] DONE %s | acc=%s | mean_lat=%ss",
                            job_index,
                            len(jobs),
                            job.job_id,
                            summary.get("accuracy") or summary.get("label_accuracy"),
                            summary.get("mean_latency_s"),
                        )
                    except Exception as exc:
                        LOGGER.error("[%d/%d] FAILED %s: %s", job_index, len(jobs), job.job_id, exc)
                        LOGGER.debug(traceback.format_exc())
                        fail = {
                            "job_id": job.job_id,
                            "status": "failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        summaries.append(fail)
                        with open(manifest_path, "a", encoding="utf-8") as mf:
                            mf.write(json.dumps(fail, ensure_ascii=False) + "\n")
            finally:
                unload_model(model)
                models_dict.clear()
    finally:
        meter.shutdown()

    # Write aggregate summary JSON
    summary_path = output_dir / "campaign_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "max_items": args.max_items,
                "n_jobs": len(jobs),
                "gpu": meter.gpu_name,
                "summaries": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    LOGGER.info("Wrote %s", summary_path)

    print_final_summary(summaries)
    n_failed = sum(1 for s in summaries if s.get("status") == "failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
