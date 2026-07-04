"""
Real model inference for full evaluation. Uses Hugging Face token from environment when set
(HF_TOKEN or HUGGING_FACE_HUB_TOKEN); never hardcode the token. Public Hub models load without
a token; gated checkpoints still require a token plus access approval on Hugging Face.

Default models (gated — set HF_TOKEN + Hub access for both):
  SLM = google/gemma-2-2b-it
  LLM = meta-llama/Llama-2-7b-chat-hf
Open Hub (no gated approval): GP_MODEL_SLM=Qwen/Qwen2.5-1.5B-Instruct
  GP_MODEL_LLM=Qwen/Qwen2.5-7B-Instruct  (or GP_BENCH_OPEN_MODELS=1 in eval_benchmarks.py)
Override at process start with GP_MODEL_SLM / GP_MODEL_LLM (before load_models).

RAG: set RAG_INDEX_DIR (folder with index.faiss + chunks.jsonl) or RAG_FAISS_INDEX +
RAG_CHUNKS_JSONL. Dense retrieval uses FAISS + RAG_EMBED_MODEL (default all-MiniLM-L6-v2).
If FAISS is missing or fails, the runner scores real passages in chunks.jsonl with lexical
overlap (rag_source=lexical) — not generic placeholder text. Discovery order: /kaggle/input,
/kaggle/working/rag_index, ./rag_index. Skips the ~24-chunk diabetes seed when a PubMed corpus
exists; seed auto-build runs only if no corpus is found (RAG_AUTO_BUILD=0 to disable).
eval_benchmarks.py can stage your dataset into working first.

4-bit loads: use bitsandbytes>=0.46.1 with your Transformers version. On Kaggle, if **bitsandbytes
is not installed at all**, this module runs a **one-shot** ``pip install`` (no prior import, so no
duplicate CUDA op issue). If a **too-old** bitsandbytes is already on the image, use **notebook Cell 1**
+ **Kernel → Restart Session**, or set ``GP_BENCH_KAGGLE_AUTO_PIP=1`` for legacy in-process upgrade
(not recommended). To block any auto-pip: ``GP_BENCH_NO_AUTO_PIP=1``.

Ablations (env, or set by eval_benchmarks.py --rag_* flags):
  RAG_TOP_K (default 3), RAG_CONTEXT_MAX_CHARS (default 4500), RAG_FORCE_MOCK=1 (skip FAISS; lexical only)
  RAG_FETCH_MULT (default 6): FAISS retrieves top_k * mult candidates before rerank/dedup.
  RAG_LEXICAL_WEIGHT (default 0.22): blend dense sim with token Jaccard vs query (0 = dense only).
  RAG_DEDUP_JACCARD (default 0.88): skip a chunk if token Jaccard with an already kept chunk >= this.
  RAG_RETRIEVAL_LEGACY=1: disable hybrid RRF/BM25/rerank (original FAISS+lexical blend).
  RAG_DISABLE_RRF=1 / RAG_DISABLE_RERANK=1: turn off RRF or cross-encoder reranker.
  RAG_RERANK_CANDIDATES (default 20), RAG_RRF_K (60), RAG_MIN_DENSE_SCORE (0.35).
  RAG_RERANKER_MODEL (default cross-encoder/ms-marco-MiniLM-L-6-v2).
  RAG_REPAIR_PROMPT=1 (default): biomedical evidence template in generate_response.
  RAG_AUTO_BUILD, RAG_AUTO_BUILD_DIR, GP_BENCH_NO_AUTO_PIP, GP_BENCH_NO_AUTO_PIP_BNB
  GP_BENCH_KAGGLE_AUTO_PIP (Kaggle: allow in-process pip upgrade when bnb exists but is too old)
  GEN_MAX_NEW_TOKENS (default 32; increase for longer PubMedQA free-text, max 256 clamp)
  GP_MODEL_SLM, GP_MODEL_LLM (local GPU model ids; see module docstring)
"""
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Instruction-tuned defaults (Gemma SLM + Llama LLM; HF_TOKEN + Hub access required).
_DEFAULT_SLM = "google/gemma-2-2b-it"
_DEFAULT_LLM = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_RAG_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_id_requires_hf_gating(model_id: str) -> bool:
    low = (model_id or "").strip().lower()
    return "meta-llama/" in low or "google/gemma" in low


def refresh_model_ids_from_env() -> None:
    """Re-read GP_MODEL_SLM / GP_MODEL_LLM (eval_benchmarks sets defaults before import)."""
    global MODEL_SLM, MODEL_LLM
    MODEL_SLM = (os.environ.get("GP_MODEL_SLM") or _DEFAULT_SLM).strip()
    MODEL_LLM = (os.environ.get("GP_MODEL_LLM") or _DEFAULT_LLM).strip()


refresh_model_ids_from_env()
_FAISS_IMPORT_ERROR: Optional[str] = None
_FAISS_PREWARMED: bool = False
_RAG_HYBRID_LOG_COUNT: int = 0

# Filled when use_rag=True so eval_benchmarks can log retrieved_context (single-threaded runs).
LAST_RAG_EVIDENCE: str = ""
LAST_RAG_SOURCE: str = ""  # "faiss" | "lexical" | "mock" | "stub" | "none"
# Per-chunk metadata: idx, source (e.g. pubmedqa_*), lexical_score, dense_score, combined_score, text_snippet.
LAST_RAG_HITS: List[Dict[str, Any]] = []
# Full ranked corpus source ids (for Recall@K / MRR); length up to RAG_METRICS_MAX_K.
LAST_RAG_RANKED_SOURCES: List[str] = []
# Per-query retrieval diagnostics (RAG repair plan Step 07).
LAST_RAG_DIAGNOSTIC: Dict[str, Any] = {}
_RAG_MOCK_FALLBACK_WARNED: bool = False
_RAG_AUTOBUILD_TRIED: bool = False
_BNB_PIP_TRIED: bool = False
_KAGGLE_PIP_HINT_KEYS: Set[str] = set()


def _get_hf_token() -> Optional[str]:
    """Read Hugging Face token from environment. Never log or store it."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _default_rag_embed_model() -> str:
    return (
        os.environ.get("RAG_EMBED_MODEL", "").strip()
        or DEFAULT_RAG_EMBED_MODEL
    )


def ensure_rag_index_env() -> str:
    """
    Pin ``RAG_INDEX_DIR`` + explicit ``RAG_FAISS_INDEX`` / ``RAG_CHUNKS_JSONL`` paths.

    Call before FAISS retrieval so a staged ``/kaggle/working/rag_index`` is always visible
    to ``_rag_paths()`` even if only the directory was set earlier.
    """
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    candidates: List[str] = []
    if d:
        candidates.append(os.path.abspath(d))
    for cand in (
        "/content/drive/MyDrive/Colab Notebooks/rag_index",
        "/content/rag_index",
        "/kaggle/working/rag_index",
        os.path.join(os.getcwd(), "rag_index"),
        _default_rag_autobuild_dir(),
    ):
        ap = os.path.abspath(cand)
        if ap not in candidates:
            candidates.append(ap)
    for ap in _iter_autodiscover_rag_dirs():
        if ap not in candidates:
            candidates.append(ap)
    for ap in candidates:
        ip, cp = _rag_pair_in_dir(ap)
        if os.path.isfile(ip) and os.path.isfile(cp) and not _is_builtin_seed_chunks(cp):
            os.environ["RAG_INDEX_DIR"] = ap
            os.environ["RAG_FAISS_INDEX"] = ip
            os.environ["RAG_CHUNKS_JSONL"] = cp
            os.environ.setdefault("RAG_EMBED_MODEL", DEFAULT_RAG_EMBED_MODEL)
            os.environ["RAG_AUTO_BUILD"] = "0"
            return ap
    return d


def _faiss_deps_available() -> bool:
    """True when faiss + numpy import; embedding uses ST or transformers fallback."""
    global _FAISS_IMPORT_ERROR
    try:
        import faiss  # type: ignore  # noqa: F401
        import numpy as np  # noqa: F401

        _ = np
        return True
    except Exception as ex:
        if _FAISS_IMPORT_ERROR is None:
            _FAISS_IMPORT_ERROR = str(ex)
        if _maybe_autopip_faiss_cpu_only():
            try:
                import faiss  # type: ignore  # noqa: F401
                import numpy as np  # noqa: F401

                _ = np
                _FAISS_IMPORT_ERROR = None
                return True
            except Exception as ex2:
                _FAISS_IMPORT_ERROR = str(ex2)
        return False


def _transformers_embed_stack_ok() -> bool:
    try:
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel  # noqa: F401

        _ = AutoModel, AutoTokenizer, PreTrainedModel
        return True
    except Exception:
        return False


class _TransformersMiniLMEmbedder:
    """Mean-pooled MiniLM encode when sentence-transformers cannot load."""

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._F = F
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        dev = (device or "cpu").strip() or "cpu"
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev
        self.model.to(self.device)

    def encode(
        self,
        sentences: List[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> Any:
        import numpy as np

        torch = self._torch
        F = self._F
        with torch.no_grad():
            batch = self.tokenizer(
                list(sentences),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            token_emb = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            summed = torch.sum(token_emb * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
            if normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)
        if convert_to_numpy:
            return emb.cpu().numpy()
        return emb


def _load_rag_embedder(embed_name: str) -> Any:
    """Object with ``encode(...)`` for FAISS query vectors."""
    device = _rag_embed_device()
    st_err: Optional[Exception] = None
    try:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(embed_name, device=device)
        except TypeError:
            return SentenceTransformer(embed_name)
    except Exception as ex:
        st_err = ex
    if not _transformers_embed_stack_ok():
        raise RuntimeError(
            f"sentence-transformers failed ({st_err}); transformers embed stack unavailable "
            "(install transformers>=4.43, accelerate, safetensors; restart kernel)."
        ) from st_err
    try:
        emb = _TransformersMiniLMEmbedder(embed_name, device)
        print(
            f"RAG: using transformers AutoModel embedder for {embed_name!r} "
            f"(sentence-transformers unavailable: {st_err})",
            flush=True,
        )
        return emb
    except Exception as tf_ex:
        raise RuntimeError(
            f"RAG embedder load failed: sentence-transformers={st_err!r}; transformers={tf_ex!r}"
        ) from tf_ex


def _rag_embed_device() -> str:
    """Default CPU so embedding works while GPU is loaded with Gemma/Llama."""
    return (os.environ.get("RAG_EMBED_DEVICE") or "cpu").strip() or "cpu"


def print_startup_diagnostics() -> None:
    """Log resolved models and RAG paths (call once before benchmarks)."""
    print(f"SLM: {MODEL_SLM}", flush=True)
    print(f"LLM: {MODEL_LLM}", flush=True)
    print(f"RAG_EMBED_MODEL: {_default_rag_embed_model()}", flush=True)
    ensure_rag_index_env()
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    if d:
        print(f"RAG_INDEX_DIR: {d}", flush=True)
    ip, cp = _rag_paths()
    if ip and cp:
        print(f"RAG index: {ip}", flush=True)
        print(f"RAG chunks: {cp}", flush=True)


def verify_rag_index(require_faiss: bool = False) -> Dict[str, Any]:
    """
    Inspect FAISS index + dependencies. Returns a JSON-serialisable status dict.
    """
    global _FAISS_IMPORT_ERROR
    out: Dict[str, Any] = {
        "faiss_deps_ok": False,
        "faiss_load_ok": False,
        "faiss_import_error": _FAISS_IMPORT_ERROR or "",
        "embed_model": _default_rag_embed_model(),
        "index_path": "",
        "chunks_path": "",
        "ntotal": 0,
        "dimension": 0,
        "index_type": "",
    }
    ensure_rag_index_env()
    idx_path, chunk_path = _rag_paths()
    out["index_path"] = idx_path or ""
    out["chunks_path"] = chunk_path or ""
    if not idx_path or not os.path.isfile(idx_path):
        if require_faiss:
            raise RuntimeError(
                "FAISS index not found. Set RAG_INDEX_DIR or pass --rag_index_dir "
                "with index.faiss + chunks.jsonl."
            )
        return out
    out["faiss_deps_ok"] = _faiss_deps_available()
    out["transformers_embed_ok"] = _transformers_embed_stack_ok()
    try:
        import faiss  # type: ignore

        index = faiss.read_index(idx_path)
        out["faiss_load_ok"] = True
        out["ntotal"] = int(index.ntotal)
        out["dimension"] = int(index.d)
        out["index_type"] = type(index).__name__
    except Exception as ex:
        _FAISS_IMPORT_ERROR = str(ex)
        out["faiss_import_error"] = str(ex)
        if require_faiss:
            raise RuntimeError(
                f"FAISS index exists but could not load: {ex}. "
                "pip install faiss-cpu sentence-transformers"
            ) from ex
    if chunk_path and os.path.isfile(chunk_path):
        try:
            out["n_chunks"] = sum(
                1 for ln in open(chunk_path, encoding="utf-8", errors="replace") if ln.strip()
            )
        except OSError:
            out["n_chunks"] = 0
    return out


def mock_retrieval(query: str, top_k: int = 3) -> str:
    """
    Legacy name kept for ``RAG_FORCE_MOCK`` ablations.

    Uses real corpus lexical retrieval when ``chunks.jsonl`` is available; never returns
    generic diabetes placeholder text.
    """
    max_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "4500"))
    block = _retrieve_lexical(query, top_k=top_k, max_context_chars=max_chars)
    if block:
        return block
    return _stub_retrieval(query, top_k=top_k)


def _rag_force_mock() -> bool:
    return os.environ.get("RAG_FORCE_MOCK", "").strip() in ("1", "true", "yes", "on")


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _env_falsy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("0", "false", "no", "off")


def _on_kaggle_working() -> bool:
    return os.path.isdir("/kaggle/working")


def _kaggle_runtime_pip_allowed() -> bool:
    """
    Kaggle often pre-imports bitsandbytes with the base image; upgrading via pip in the same
    process can register duplicate CUDA ops. Runtime pip is opt-in on Kaggle only.
    """
    if not _on_kaggle_working():
        return True
    return _env_truthy("GP_BENCH_KAGGLE_AUTO_PIP")


def _print_kaggle_install_hint(key: str, what: str) -> None:
    if key in _KAGGLE_PIP_HINT_KEYS:
        return
    _KAGGLE_PIP_HINT_KEYS.add(key)
    print(
        f"GP_BENCH: skipping in-process pip for {what} on Kaggle when the package exists but is "
        "too old or unusable (upgrading a loaded bitsandbytes can duplicate CUDA op registration). "
        "Install in the first notebook cell, then Kernel → Restart Session, or set "
        "GP_BENCH_KAGGLE_AUTO_PIP=1 for a forced in-process upgrade (not recommended).",
        flush=True,
    )


def _bitsandbytes_package_missing() -> bool:
    """True if ``import bitsandbytes`` fails (wheel not installed)."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return True
    return False


def _run_pip_bitsandbytes() -> None:
    import subprocess
    import sys

    print(
        "GP_BENCH: pip install -q -U 'bitsandbytes>=0.46.1' (Transformers 4-bit minimum)...",
        flush=True,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-U",
                "bitsandbytes>=0.46.1",
            ],
            check=False,
            timeout=600,
        )
    except Exception as ex:
        print(f"GP_BENCH: bitsandbytes auto-install failed: {ex}", flush=True)
    _invalidate_bitsandbytes_modules()


def _maybe_autopip_bitsandbytes() -> None:
    """
    Ensure bitsandbytes when 4-bit is requested. Local / Kaggle+opt-in: pip if still not OK.
    Kaggle default: pip only if the package is **missing** (ImportError); if an old build is
    present, print the Cell 1 + restart hint instead of upgrading in-process.
    """
    global _BNB_PIP_TRIED
    if _BNB_PIP_TRIED:
        return
    if _env_truthy("GP_BENCH_NO_AUTO_PIP") or _env_truthy("GP_BENCH_NO_AUTO_PIP_BNB"):
        return
    import torch

    if not torch.cuda.is_available():
        return
    if _bitsandbytes_import_ok():
        return

    on_kaggle = _on_kaggle_working()
    kaggle_pip = _kaggle_runtime_pip_allowed()
    missing = _bitsandbytes_package_missing()

    if on_kaggle and not kaggle_pip:
        if missing:
            print(
                "GP_BENCH: bitsandbytes not installed; running one-shot pip (no prior bnb import in "
                "this process). After install, 4-bit load is retried.",
                flush=True,
            )
            _BNB_PIP_TRIED = True
            _run_pip_bitsandbytes()
        else:
            if not _bitsandbytes_import_ok():
                _print_kaggle_install_hint("bnb", "bitsandbytes>=0.46.1")
            _BNB_PIP_TRIED = True
        return

    _BNB_PIP_TRIED = True
    _run_pip_bitsandbytes()


def _bnb_version_tuple() -> Optional[Tuple[int, int, int]]:
    """Parse bitsandbytes version for comparison (coarse; ignores pre-release suffixes)."""
    try:
        import importlib.metadata as im

        raw = im.version("bitsandbytes").split("+")[0].strip()
        parts = raw.split(".")
        nums: List[int] = []
        for p in parts[:3]:
            acc = ""
            for c in p:
                if c.isdigit():
                    acc += c
                else:
                    break
            nums.append(int(acc) if acc else 0)
        while len(nums) < 3:
            nums.append(0)
        return (nums[0], nums[1], nums[2])
    except Exception:
        return None


def _bitsandbytes_import_ok() -> bool:
    """
    True only if bitsandbytes is importable, meets Transformers' minimum version, and
    transformers reports it as usable (avoids selecting 4-bit then failing in from_pretrained).
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    vt = _bnb_version_tuple()
    if vt is not None and vt < (0, 46, 1):
        return False
    try:
        from transformers.utils.import_utils import is_bitsandbytes_available

        return bool(is_bitsandbytes_available())
    except Exception:
        return False


def _invalidate_bitsandbytes_modules() -> None:
    try:
        import sys

        for k in list(sys.modules):
            if k == "bitsandbytes" or k.startswith("bitsandbytes."):
                del sys.modules[k]
    except Exception:
        pass


def _hf_stack_load_hint(exc: BaseException) -> str:
    low = str(exc).lower()
    if "pretrainedmodel" in low or "gemma2" in low or "could not import module" in low:
        return (
            " Fix: Kernel → Restart Session, then "
            "pip install -q 'transformers>=4.43.0,<5' 'huggingface-hub>=0.23' "
            "'accelerate>=0.26' safetensors sentence-transformers --upgrade "
            "(after numpy/scipy/pandas Cell 1 + restart). "
            "Set GP_BENCH_SKIP_RESTART_CHECK=1 only after restart."
        )
    return ""


def _from_pretrained_with_dtype(model_cls: Any, model_name: str, model_dtype: Any, **load_kwargs: Any) -> Any:
    """``from_pretrained`` using ``dtype`` (``torch_dtype`` is deprecated in recent transformers)."""
    try:
        return model_cls.from_pretrained(model_name, dtype=model_dtype, **load_kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=model_dtype, **load_kwargs)


def _from_pretrained_causal_lm_safe(
    model_name: str,
    model_dtype: Any,
    bnb_kwargs: Dict[str, Any],
    fp_kwargs: Dict[str, Any],
    kwargs: Dict[str, Any],
    *,
    use_4bit_cuda: bool,
    device: str,
) -> Any:
    """Load causal LM; if 4-bit fails at runtime, fall back to fp16 (higher VRAM)."""
    from transformers import AutoModelForCausalLM

    def _load(load_kwargs: Dict[str, Any]) -> Any:
        return _from_pretrained_with_dtype(
            AutoModelForCausalLM,
            model_name,
            model_dtype,
            **load_kwargs,
            **kwargs,
        )

    try:
        return _load(bnb_kwargs)
    except Exception as ex:
        if (
            use_4bit_cuda
            and device == "cuda"
            and bnb_kwargs.get("quantization_config") is not None
        ):
            low = str(ex).lower()
            if "bitsandbytes" in low or "4-bit" in low or "bnb" in low or "quantization" in low:
                print(
                    f"GP_BENCH: 4-bit load failed ({ex!r}); retrying fp16 (higher VRAM).",
                    flush=True,
                )
                return _load(fp_kwargs)
        hint = _hf_stack_load_hint(ex)
        if hint:
            raise RuntimeError(f"{ex}.{hint}") from ex
        raise


def _rag_autobuild_enabled() -> bool:
    if _env_falsy("RAG_AUTO_BUILD"):
        return False
    if _env_truthy("RAG_AUTO_BUILD"):
        return True
    return _on_kaggle_working()


def _default_rag_autobuild_dir() -> str:
    d = os.environ.get("RAG_AUTO_BUILD_DIR", "").strip()
    if d:
        return d
    if _on_kaggle_working():
        return "/kaggle/working/rag_index"
    return os.path.abspath(os.path.join(os.getcwd(), "rag_index"))


_SEED_RAG_TEXTS: Tuple[str, ...] = (
    "Metformin is commonly recommended as first-line pharmacotherapy for type 2 diabetes when not contraindicated.",
    "SGLT2 inhibitors may reduce heart failure hospitalizations in patients with HFrEF or HFpEF.",
    "Hypertension targets are individualized; many guidelines suggest <130/80 mmHg for adults at elevated risk.",
    "Statin therapy lowers LDL-C and reduces atherosclerotic cardiovascular risk in appropriate primary and secondary prevention.",
    "Pneumococcal vaccination schedules depend on age, risk factors, and prior immunization history.",
    "Acute coronary syndrome management includes antiplatelet therapy, anticoagulation when indicated, and timely revascularization strategies.",
    "Asthma step-up therapy follows inhaled corticosteroid dose and add-on LABA per control and exacerbation frequency.",
    "Chronic kidney disease staging uses eGFR and urine albumin-to-creatinine ratio to guide monitoring and medication dosing.",
    "Anticoagulation for atrial fibrillation balances stroke reduction with bleeding risk scores and patient preferences.",
    "Sepsis bundles emphasize early cultures, broad antibiotics after cultures when feasible, and hemodynamic resuscitation.",
    "Depression treatment combines psychotherapy and pharmacotherapy selection based on symptoms, comorbidities, and prior response.",
    "Osteoporosis management includes calcium and vitamin D adequacy, fall risk reduction, and pharmacologic therapy when indicated.",
    "Migraine preventive options include beta-blockers, antiepileptics, and CGRP pathway inhibitors depending on comorbidities.",
    "COPD exacerbations may warrant bronchodilators, corticosteroids, antibiotics when bacterial infection suspected, and oxygen titration.",
    "Thyroid function tests interpret TSH with free T4 in context of pregnancy, pituitary disease, and medications.",
    "Hepatitis B screening and vaccination are recommended for at-risk populations per public health guidance.",
    "HIV pre-exposure prophylaxis requires adherence monitoring and periodic testing for HIV and other STIs.",
    "Rheumatoid arthritis treatment uses DMARDs early; biologics are considered when inadequate response or contraindications.",
    "Inflammatory bowel disease therapy escalates from aminosalicylates to immunomodulators and biologics based on severity.",
    "Stroke evaluation distinguishes ischemic versus hemorrhagic etiology before reperfusion or blood pressure goals.",
    "Pediatric fever evaluation depends on age, immunization status, focal findings, and toxic appearance.",
    "Travel medicine includes destination-specific vaccines, malaria chemoprophylaxis, and traveler diarrhea counseling.",
    "Pain management multimodal regimens reduce opioid exposure while addressing functional goals.",
    "Ethics in clinical trials emphasize informed consent, equipoise, and independent safety monitoring.",
)


def _maybe_autopip_faiss_cpu_only() -> bool:
    """Install only faiss-cpu (safe on Kaggle in a fresh ``!python`` process)."""
    if _env_truthy("GP_BENCH_NO_AUTO_PIP"):
        return False
    try:
        import faiss  # type: ignore  # noqa: F401

        return True
    except ImportError:
        pass
    import subprocess
    import sys

    print("GP_BENCH: pip install -q faiss-cpu (dense RAG)...", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"],
            check=False,
            timeout=300,
        )
        import faiss  # type: ignore  # noqa: F401

        return True
    except Exception as ex:
        print(f"GP_BENCH: faiss-cpu auto-install failed: {ex}", flush=True)
        return False


def _maybe_autopip_faiss_stack() -> bool:
    if _env_truthy("GP_BENCH_NO_AUTO_PIP"):
        return False
    if not _on_kaggle_working():
        return True
    if _maybe_autopip_faiss_cpu_only():
        if not _kaggle_runtime_pip_allowed():
            return True
    elif not _kaggle_runtime_pip_allowed():
        _print_kaggle_install_hint("faiss", "faiss-cpu and sentence-transformers")
        return False
    if not _kaggle_runtime_pip_allowed():
        return True
    import subprocess
    import sys

    print(
        "GP_BENCH: pip install -q sentence-transformers safetensors accelerate "
        "(optional RAG embed stack; set GP_BENCH_KAGGLE_AUTO_PIP=1)...",
        flush=True,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers>=4.43.0,<5",
                "accelerate>=0.26.0",
                "sentence-transformers>=2.2.2",
                "safetensors>=0.4.0",
            ],
            check=False,
            timeout=600,
        )
        return True
    except Exception as ex:
        print(f"GP_BENCH: sentence-transformers auto-install failed: {ex}", flush=True)
        return False


def _build_seed_faiss_index(out_dir: str) -> bool:
    """Write chunks.jsonl + index.faiss from built-in seed texts. Returns True on success."""
    import json

    try:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        if not _maybe_autopip_faiss_stack():
            return False
        try:
            import faiss  # type: ignore
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return False

    os.makedirs(out_dir, exist_ok=True)
    chunk_path = os.path.join(out_dir, "chunks.jsonl")
    rows = [{"text": t} for t in _SEED_RAG_TEXTS]
    with open(chunk_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    embed_name = (
        os.environ.get("RAG_EMBED_MODEL", "").strip()
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    st = SentenceTransformer(embed_name)
    texts = [str(r["text"]).strip() for r in rows]
    dim = st.get_sentence_embedding_dimension()
    try:
        mat = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        mat = st.encode(texts, convert_to_numpy=True)
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    idx_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    return True


def _discover_best_chunks_path() -> str:
    """Find the largest non-seed ``chunks.jsonl`` under Kaggle input / working / cwd."""
    candidates: List[Tuple[int, str]] = []

    def consider(path: str) -> None:
        if not os.path.isfile(path) or os.path.getsize(path) <= 32:
            return
        if _is_builtin_seed_chunks(path):
            return
        try:
            n_lines = sum(1 for ln in open(path, encoding="utf-8", errors="replace") if ln.strip())
        except OSError:
            return
        candidates.append((n_lines, path))

    env_cp = os.environ.get("RAG_CHUNKS_JSONL", "").strip()
    if env_cp:
        consider(env_cp)
    for cand in _iter_autodiscover_rag_dirs():
        consider(os.path.join(cand, "chunks.jsonl"))
    for root in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "chunks.jsonl" not in filenames:
                continue
            consider(os.path.join(dirpath, "chunks.jsonl"))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _try_autobuild_rag_index_once() -> None:
    global _RAG_AUTOBUILD_TRIED
    if _RAG_AUTOBUILD_TRIED:
        return
    if _rag_force_mock() or not _rag_autobuild_enabled():
        _RAG_AUTOBUILD_TRIED = True
        return
    real_cp = _corpus_chunks_path() or _discover_best_chunks_path()
    if real_cp and os.path.isfile(real_cp) and not _is_builtin_seed_chunks(real_cp):
        _RAG_AUTOBUILD_TRIED = True
        return
    ip, cp = _rag_paths()
    if ip and cp and os.path.isfile(ip) and os.path.isfile(cp):
        if not _is_builtin_seed_chunks(cp):
            _RAG_AUTOBUILD_TRIED = True
            return
    if os.environ.get("RAG_AUTO_BUILD", "").strip().lower() in ("0", "false", "no", "off"):
        _RAG_AUTOBUILD_TRIED = True
        return
    _RAG_AUTOBUILD_TRIED = True
    target = os.path.abspath(_default_rag_autobuild_dir())
    if _has_rag_pair_anywhere_except_seed():
        _RAG_AUTOBUILD_TRIED = True
        return
    if not _build_seed_faiss_index(target):
        return
    os.environ["RAG_INDEX_DIR"] = target
    _clear_rag_cache()
    print(
        f"RAG: auto-built seed FAISS index at {target} (disable with RAG_AUTO_BUILD=0; replace with build_rag_index.py for real corpus).",
        flush=True,
    )


def _rag_pair_in_dir(d: str) -> Tuple[str, str]:
    return os.path.join(d, "index.faiss"), os.path.join(d, "chunks.jsonl")


def _is_builtin_seed_chunks(chunks_path: str) -> bool:
    """True for the ~24-line diabetes/guideline seed index from _build_seed_faiss_index."""
    try:
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
                        import json

                        first_text = str(json.loads(line).get("text") or "")[:300]
                    except json.JSONDecodeError:
                        first_text = line[:300]
        if n_lines >= 100 or pubmed_hits >= 20:
            return False
        return n_lines <= 30 and (
            "metformin is commonly recommended" in first_text.lower() or pubmed_hits == 0
        )
    except OSError:
        return False


def _walk_rag_dirs(root: str, max_depth: int = 4) -> List[str]:
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


def _has_rag_pair_anywhere_except_seed() -> bool:
    for cand in _iter_autodiscover_rag_dirs():
        cp = os.path.join(cand, "chunks.jsonl")
        if os.path.isfile(cp) and not _is_builtin_seed_chunks(cp):
            return True
    return False


def _iter_autodiscover_rag_dirs() -> List[str]:
    """Search order: Kaggle input (real corpus) before working seed index."""
    scored: List[Tuple[int, int, str]] = []
    seen: set[str] = set()

    def consider(dirpath: str, priority: int) -> None:
        ap = os.path.abspath(os.path.normpath(dirpath))
        if ap in seen:
            return
        ip, cp = _rag_pair_in_dir(ap)
        if not (os.path.isfile(ip) and os.path.isfile(cp)):
            return
        seen.add(ap)
        is_seed = 1 if _is_builtin_seed_chunks(cp) else 0
        try:
            n_lines = sum(1 for ln in open(cp, encoding="utf-8", errors="replace") if ln.strip())
        except OSError:
            n_lines = 0
        scored.append((priority, -n_lines, is_seed, ap))

    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            root = os.path.join(kin, name)
            for d in _walk_rag_dirs(root):
                consider(d, priority=0)

    if os.path.isdir("/kaggle"):
        consider("/kaggle/working/rag_index", priority=1)
    consider(os.path.join(os.getcwd(), "rag_index"), priority=2)

    scored.sort(key=lambda x: (x[0], x[2], x[1]))
    return [x[3] for x in scored]


def _rag_paths() -> Tuple[str, str]:
    """Return (faiss_path, chunks_jsonl_path) or ("", "")."""
    idx = os.environ.get("RAG_FAISS_INDEX", "").strip()
    chunks = os.environ.get("RAG_CHUNKS_JSONL", "").strip()
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    if d and (not idx or not chunks):
        idx, chunks = _rag_pair_in_dir(d)
    if idx and chunks and os.path.isfile(idx) and os.path.isfile(chunks):
        if not _is_builtin_seed_chunks(chunks):
            return idx, chunks
        # Env points at seed; keep searching for a real corpus unless nothing else exists.
        env_seed = (idx, chunks)
    else:
        env_seed = None

    for cand in _iter_autodiscover_rag_dirs():
        ip, cp = _rag_pair_in_dir(cand)
        if os.path.isfile(ip) and os.path.isfile(cp) and not _is_builtin_seed_chunks(cp):
            print(
                f"RAG: auto-discovered FAISS index in {cand} "
                "(set RAG_INDEX_DIR or --rag_index_dir to pin a path).",
                flush=True,
            )
            os.environ["RAG_INDEX_DIR"] = cand
            return ip, cp

    if env_seed:
        return env_seed
    for cand in _iter_autodiscover_rag_dirs():
        ip, cp = _rag_pair_in_dir(cand)
        if os.path.isfile(ip) and os.path.isfile(cp):
            print(
                f"RAG: using seed/builtin FAISS index in {cand} "
                "(replace with build_rag_index.py output for PubMed corpus).",
                flush=True,
            )
            return ip, cp
    return idx, chunks


_rag_singleton: Dict[str, Any] = {}
_RERANKER_CE_CACHE: Dict[str, Any] = {}


def get_cached_cross_encoder(model_name: str) -> Any:
    """Load ms-marco (or RAG_RERANKER_MODEL) once; reused for every hybrid+rerank query."""
    name = (model_name or "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if name in _RERANKER_CE_CACHE:
        return _RERANKER_CE_CACHE[name]
    from sentence_transformers import CrossEncoder

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    ce = CrossEncoder(name, max_length=512, device="cpu")
    _RERANKER_CE_CACHE[name] = ce
    print(f"RAG: reranker loaded and cached ({name}).", flush=True)
    return ce


def clear_reranker_cache() -> None:
    _RERANKER_CE_CACHE.clear()


def _clear_rag_cache() -> None:
    global _RAG_MOCK_FALLBACK_WARNED, LAST_RAG_HITS, LAST_RAG_RANKED_SOURCES, LAST_RAG_DIAGNOSTIC
    _rag_singleton.clear()
    _RAG_MOCK_FALLBACK_WARNED = False
    LAST_RAG_HITS = []
    LAST_RAG_RANKED_SOURCES = []
    LAST_RAG_DIAGNOSTIC = {}
    clear_reranker_cache()
    try:
        from rag_retrieval import clear_retrieval_caches

        clear_retrieval_caches()
    except ImportError:
        pass


def _rag_metrics_max_k() -> int:
    raw = os.environ.get("RAG_METRICS_MAX_K", "50").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _set_last_ranked_sources(hits: List[Dict[str, Any]]) -> None:
    """Store ordered source ids from ranked hit dicts (pre-dedup retrieval ranking)."""
    global LAST_RAG_RANKED_SOURCES
    cap = _rag_metrics_max_k()
    ranked: List[str] = []
    seen: Set[str] = set()
    for h in hits:
        src = str(h.get("source") or "").strip()
        if not src:
            idx = h.get("idx")
            if idx is not None and int(idx) >= 0:
                src = f"idx_{int(idx)}"
        if not src or src in seen:
            continue
        seen.add(src)
        ranked.append(src)
        if len(ranked) >= cap:
            break
    LAST_RAG_RANKED_SOURCES = ranked


def _set_last_rag_hits(hits: List[Dict[str, Any]]) -> None:
    global LAST_RAG_HITS
    LAST_RAG_HITS = hits


def _chunk_source_at(idx: int) -> str:
    sources: List[str] = _rag_singleton.get("sources") or []
    if 0 <= idx < len(sources):
        return str(sources[idx] or "")
    return ""


def _make_rag_hit(
    rank: int,
    idx: int,
    body: str,
    *,
    source: str = "",
    lexical_score: Optional[float] = None,
    dense_score: Optional[float] = None,
    combined_score: Optional[float] = None,
) -> Dict[str, Any]:
    src = source or _chunk_source_at(idx)
    hit: Dict[str, Any] = {
        "rank": rank,
        "idx": int(idx),
        "source": src,
        "text_snippet": (body or "")[:500],
    }
    if lexical_score is not None:
        hit["lexical_score"] = round(float(lexical_score), 6)
    if dense_score is not None:
        hit["dense_score"] = round(float(dense_score), 6)
    if combined_score is not None:
        hit["combined_score"] = round(float(combined_score), 6)
    return hit


def _load_chunk_texts(path: str) -> List[str]:
    return [str(r.get("text") or "") for r in _load_chunk_records(path)]


def _load_chunk_records(path: str) -> List[Dict[str, Any]]:
    import json

    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
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


def _corpus_chunks_path() -> str:
    """Path to chunks.jsonl (with or without a loadable FAISS index)."""
    _, cp = _rag_paths()
    if cp and os.path.isfile(cp) and not _is_builtin_seed_chunks(cp):
        return cp
    env_cp = os.environ.get("RAG_CHUNKS_JSONL", "").strip()
    if env_cp and os.path.isfile(env_cp) and not _is_builtin_seed_chunks(env_cp):
        return env_cp
    for cand in _iter_autodiscover_rag_dirs():
        p = os.path.join(cand, "chunks.jsonl")
        if os.path.isfile(p) and os.path.getsize(p) > 32 and not _is_builtin_seed_chunks(p):
            return p
    return _discover_best_chunks_path()


def _ensure_corpus_loaded() -> bool:
    """Load chunk texts + sources into ``_rag_singleton`` for lexical / FAISS retrieval."""
    chunk_path = _corpus_chunks_path()
    if not chunk_path:
        return False
    cache_key = f"corpus|{chunk_path}"
    if _rag_singleton.get("corpus_key") == cache_key and _rag_singleton.get("texts"):
        return True
    records = _load_chunk_records(chunk_path)
    texts = [str(r.get("text") or "") for r in records]
    sources = [str(r.get("source") or "") for r in records]
    if not texts:
        return False
    _rag_singleton["corpus_key"] = cache_key
    _rag_singleton["texts"] = texts
    _rag_singleton["sources"] = sources
    _rag_singleton["chunk_path"] = chunk_path
    print(
        f"RAG: loaded {len(texts)} corpus passages from {chunk_path} (lexical fallback enabled).",
        flush=True,
    )
    return True


def _chunk_header_line(rank: int, hit: Dict[str, Any]) -> str:
    meta: List[str] = [f"idx={int(hit['idx'])}"]
    if hit.get("lexical_score") is not None:
        meta.append(f"lexical={float(hit['lexical_score']):.4f}")
    if hit.get("dense_score") is not None:
        meta.append(f"dense={float(hit['dense_score']):.4f}")
    if hit.get("combined_score") is not None:
        meta.append(f"combined={float(hit['combined_score']):.4f}")
    src = str(hit.get("source") or "")
    if src:
        meta.append(f"source={src}")
    return f"Chunk {rank} ({', '.join(meta)}):"


def _format_retrieved_evidence(
    hits: List[Dict[str, Any]],
    max_context_chars: int,
    header: str,
) -> Optional[str]:
    lines: List[str] = []
    used = 0
    kept_hits: List[Dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        body = str(hit.get("body") or "")
        chunk_line = f"{_chunk_header_line(rank, hit)}\n{body}"
        if used + len(chunk_line) + 2 > max_context_chars:
            remain = max_context_chars - used - 80
            if remain > 80:
                chunk_line = f"{_chunk_header_line(rank, hit)}\n{body[:remain]}..."
                hit = dict(hit)
                hit["body"] = body[:remain] + "..."
            else:
                break
        lines.append(chunk_line)
        kept_hits.append(hit)
        used += len(chunk_line) + 2
    if not lines:
        return None
    out_hits = []
    for rank, hit in enumerate(kept_hits, start=1):
        out_hits.append(
            _make_rag_hit(
                rank,
                int(hit["idx"]),
                str(hit.get("body") or ""),
                source=str(hit.get("source") or ""),
                lexical_score=hit.get("lexical_score"),
                dense_score=hit.get("dense_score"),
                combined_score=hit.get("combined_score"),
            )
        )
    _set_last_rag_hits(out_hits)
    return header % len(lines) + "\n\n".join(lines)


def _lexical_score(query: str, body: str, q_toks: Optional[Set[str]] = None) -> float:
    q_toks = q_toks or _rag_token_set(query)
    if not q_toks:
        return 0.0
    body = (body or "").strip()
    if not body:
        return 0.0
    ql = (query or "").lower()
    toks = _rag_token_set(body)
    jac = _rag_token_jaccard(q_toks, toks)
    overlap = len(q_toks & toks)
    phrase_bonus = 0.0
    for t in sorted(q_toks, key=len, reverse=True):
        if len(t) >= 5 and t in ql and t in body.lower():
            phrase_bonus += 0.02
    return jac + 0.08 * min(overlap, 40) + phrase_bonus


def _rank_chunks_lexical(query: str, texts: List[str]) -> List[Tuple[float, int, str]]:
    q_toks = _rag_token_set(query)
    if not q_toks:
        return []
    scored: List[Tuple[float, int, str]] = []
    for i, raw in enumerate(texts):
        body = (raw or "").strip()
        if not body:
            continue
        scored.append((_lexical_score(query, body, q_toks), i, body))
    scored.sort(key=lambda x: (-x[0], -len(x[2])))
    return scored


def _retrieve_lexical(query: str, top_k: int, max_context_chars: int) -> Optional[str]:
    """
    Question-specific retrieval over ``chunks.jsonl`` without FAISS (real PubMed/MedQA passages).
    """
    if not _ensure_corpus_loaded():
        return None
    texts: List[str] = _rag_singleton["texts"]
    scored = _rank_chunks_lexical(query, texts)
    if not scored:
        return None
    metrics_hits = [
        {
            "idx": idx,
            "source": _chunk_source_at(idx),
            "lexical_score": lex_score,
        }
        for lex_score, idx, _body in scored[: _rag_metrics_max_k()]
    ]
    _set_last_ranked_sources(metrics_hits)
    fetch_mult = int(os.environ.get("RAG_FETCH_MULT", "6"))
    fetch_mult = max(1, fetch_mult)
    fetch_n = min(len(scored), max(int(top_k) * fetch_mult, int(top_k) + 4))
    dedup_thr = float(os.environ.get("RAG_DEDUP_JACCARD", "0.88"))
    dedup_thr = max(0.0, min(1.0, dedup_thr))

    selected: List[Dict[str, Any]] = []
    kept_tok_sets: List[Set[str]] = []
    for lex_score, idx, body in scored[:fetch_n]:
        if len(selected) >= int(top_k):
            break
        toks = _rag_token_set(body)
        if kept_tok_sets and any(_rag_token_jaccard(toks, kt) >= dedup_thr for kt in kept_tok_sets):
            continue
        selected.append(
            {
                "idx": idx,
                "body": body,
                "source": _chunk_source_at(idx),
                "lexical_score": lex_score,
            }
        )
        kept_tok_sets.append(toks)

    return _format_retrieved_evidence(
        selected,
        max_context_chars,
        "RETRIEVED EVIDENCE (corpus lexical top-%d):\n\n",
    )


def _stub_retrieval(query: str, top_k: int = 3) -> str:
    """Last resort when no medical corpus file is available (not fabricated guidelines)."""
    q = (query or "").strip()
    excerpt = q[: min(1200, len(q))]
    _set_last_rag_hits(
        [
            {
                "rank": 1,
                "idx": -1,
                "source": "",
                "lexical_score": 0.0,
                "text_snippet": excerpt[:500],
            }
        ]
    )
    _set_last_ranked_sources([])
    return (
        f"RETRIEVED EVIDENCE (no corpus — query context only, {top_k} slots):\n\n"
        f"Chunk 1 (idx=-1, lexical=0.0000, source=):\n{excerpt}\n\n"
        "Note: Build chunks.jsonl from PubMedQA/MedQA abstracts "
        "(see build_rag_index.py or eval_benchmarks predictions export)."
    )


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)


def _rag_token_set(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _rag_token_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _load_faiss_index_into_singleton() -> bool:
    """Load FAISS + chunks + embedder into ``_rag_singleton``. Returns False if unavailable."""
    global _rag_singleton, _FAISS_IMPORT_ERROR
    ensure_rag_index_env()
    idx_path, chunk_path = _rag_paths()
    if not idx_path or not chunk_path or not os.path.isfile(idx_path) or not os.path.isfile(chunk_path):
        if not _FAISS_IMPORT_ERROR:
            _FAISS_IMPORT_ERROR = "missing index.faiss or chunks.jsonl (set RAG_INDEX_DIR)"
        return False
    if not _faiss_deps_available():
        return False

    try:
        import faiss  # type: ignore
    except Exception as ex:
        if _FAISS_IMPORT_ERROR is None:
            _FAISS_IMPORT_ERROR = str(ex)
        return False

    embed_name = _default_rag_embed_model()
    cache_key = f"{idx_path}|{chunk_path}|{embed_name}|{_rag_embed_device()}"
    if _rag_singleton.get("key") == cache_key and _rag_singleton.get("index") is not None:
        return True

    clear_reranker_cache()
    try:
        from rag_retrieval import clear_retrieval_caches

        clear_retrieval_caches()
    except ImportError:
        pass
    _rag_singleton.clear()
    _rag_singleton["key"] = cache_key
    try:
        st = _load_rag_embedder(embed_name)
    except Exception as ex:
        if _FAISS_IMPORT_ERROR is None:
            _FAISS_IMPORT_ERROR = str(ex)
        return False
    index = faiss.read_index(idx_path)
    records = _load_chunk_records(chunk_path)
    texts = [str(r.get("text") or "") for r in records]
    sources = [str(r.get("source") or "") for r in records]
    if index.ntotal != len(texts):
        print(
            f"RAG warning: FAISS ntotal={index.ntotal} != len(chunks)={len(texts)}; "
            "truncating/padding to match (check build_rag_index.py).",
            flush=True,
        )
        if len(texts) > index.ntotal:
            texts = texts[: index.ntotal]
            sources = sources[: index.ntotal]
        else:
            pad = index.ntotal - len(texts)
            texts.extend([""] * pad)
            sources.extend([""] * pad)
    _rag_singleton["index"] = index
    _rag_singleton["texts"] = texts
    _rag_singleton["sources"] = sources
    _rag_singleton["st"] = st
    print(
        f"RAG: loaded FAISS index ({index.ntotal} vectors, d={index.d}) + chunks from {chunk_path} "
        f"(embed={embed_name!r}, device={_rag_embed_device()})",
        flush=True,
    )
    return True


def prewarm_faiss_index() -> bool:
    """Load FAISS once before the benchmark loop so retrieval uses semantic search."""
    global _FAISS_PREWARMED
    ensure_rag_index_env()
    ok = _load_faiss_index_into_singleton()
    _FAISS_PREWARMED = ok
    if ok:
        print("RAG: FAISS prewarmed — rag_source should be 'faiss' for RAG rows.", flush=True)
        if not _env_truthy("RAG_RETRIEVAL_LEGACY") and not _env_truthy("RAG_DISABLE_RERANK"):
            try:
                top_k = int(os.environ.get("RAG_TOP_K", "3"))
                max_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "2000"))
                _retrieve_faiss_hybrid(
                    "warmup biomedical retrieval query for reranker cache",
                    top_k=top_k,
                    max_context_chars=max_chars,
                )
                print("RAG: hybrid reranker prewarmed (cross-encoder cached).", flush=True)
            except Exception as ex:
                print(f"RAG: reranker prewarm skipped ({ex})", flush=True)
    elif _FAISS_IMPORT_ERROR:
        print(
            f"RAG: FAISS prewarm failed ({_FAISS_IMPORT_ERROR}). "
            "pip install faiss-cpu 'transformers>=4.43,<5' 'huggingface-hub>=0.23' accelerate "
            "sentence-transformers safetensors; set RAG_INDEX_DIR; restart kernel.",
            flush=True,
        )
    return ok


def _retrieve_faiss_hybrid(query: str, top_k: int, max_context_chars: int) -> Optional[str]:
    """RRF dense+BM25 + cross-encoder rerank (RAG repair plan Steps 04–06)."""
    global LAST_RAG_DIAGNOSTIC
    if not _load_faiss_index_into_singleton():
        return None
    index = _rag_singleton["index"]
    texts: List[str] = _rag_singleton["texts"]
    sources: List[str] = _rag_singleton.get("sources") or [""] * len(texts)
    embedder = _rag_singleton["st"]
    try:
        from rag_retrieval import hybrid_retrieve
    except ImportError:
        return None
    block, final_hits, ranked_metrics, method, diag = hybrid_retrieve(
        query,
        index=index,
        texts=texts,
        sources=sources,
        embedder=embedder,
        top_k=int(top_k),
        max_context_chars=max_context_chars,
        metrics_max_k=_rag_metrics_max_k(),
    )
    LAST_RAG_DIAGNOSTIC = dict(diag)
    if ranked_metrics:
        _set_last_ranked_sources(ranked_metrics)
    if final_hits:
        out_hits = []
        for h in final_hits:
            out_hits.append(
                _make_rag_hit(
                    int(h.get("rank") or 0),
                    int(h["idx"]),
                    str(h.get("body") or ""),
                    source=str(h.get("source") or ""),
                    lexical_score=h.get("bm25_score"),
                    dense_score=h.get("dense_score"),
                    combined_score=h.get("reranker_score"),
                )
            )
        _set_last_rag_hits(out_hits)
    if block and not _env_truthy("RAG_QUIET"):
        global _RAG_HYBRID_LOG_COUNT
        _RAG_HYBRID_LOG_COUNT += 1
        if _RAG_HYBRID_LOG_COUNT <= 3 or _env_truthy("RAG_VERBOSE"):
            print(
                f"RAG: {method} retrieval in {diag.get('retrieval_latency_ms', '?')} ms, "
                f"context={diag.get('context_char_length', 0)} chars",
                flush=True,
            )
        elif _RAG_HYBRID_LOG_COUNT == 4:
            print("RAG: further hybrid timing lines suppressed (set RAG_VERBOSE=1 to log all).", flush=True)
    return block


def _retrieve_faiss(query: str, top_k: int, max_context_chars: int) -> Optional[str]:
    """
    Return formatted evidence string, or None if unavailable / error.
    """
    global _rag_singleton, _FAISS_IMPORT_ERROR, LAST_RAG_DIAGNOSTIC
    if not _load_faiss_index_into_singleton():
        if _FAISS_IMPORT_ERROR and not getattr(_retrieve_faiss, "_warned", False):
            _retrieve_faiss._warned = True  # type: ignore[attr-defined]
            print(f"RAG: FAISS skipped — {_FAISS_IMPORT_ERROR}", flush=True)
        return None

    import faiss  # type: ignore
    import numpy as np

    index = _rag_singleton["index"]
    texts: List[str] = _rag_singleton["texts"]
    embedder = _rag_singleton["st"]

    try:
        try:
            qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            qv = embedder.encode([query], convert_to_numpy=True)
        if qv.dtype != np.float32:
            qv = qv.astype(np.float32)
        faiss.normalize_L2(qv)
        ntotal = int(index.ntotal)
        k_legacy = min(int(top_k), ntotal, max(1, ntotal))

        if not _env_truthy("RAG_RETRIEVAL_LEGACY"):
            hybrid_block = _retrieve_faiss_hybrid(query, top_k, max_context_chars)
            if hybrid_block:
                return hybrid_block

        q_toks = _rag_token_set(query)
        if _env_truthy("RAG_RETRIEVAL_LEGACY"):
            sims, ids = index.search(qv, k_legacy)
            legacy_hits: List[Dict[str, Any]] = []
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
                legacy_hits.append(
                    {
                        "idx": jj,
                        "body": body,
                        "source": _chunk_source_at(jj),
                        "dense_score": dense,
                        "lexical_score": _lexical_score(query, body, q_toks),
                    }
                )
                if len(legacy_hits) >= int(top_k):
                    break
            _set_last_ranked_sources(legacy_hits)
            return _format_retrieved_evidence(
                legacy_hits, max_context_chars, "RETRIEVED EVIDENCE (FAISS top-%d):\n\n"
            )

        fetch_mult = int(os.environ.get("RAG_FETCH_MULT", "6"))
        fetch_mult = max(1, fetch_mult)
        fetch_k = min(max(int(top_k) * fetch_mult, int(top_k) + 4), ntotal, max(1, ntotal))
        w_lex = float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.22"))
        w_lex = max(0.0, min(1.0, w_lex))
        dedup_thr = float(os.environ.get("RAG_DEDUP_JACCARD", "0.88"))
        dedup_thr = max(0.0, min(1.0, dedup_thr))

        sims, ids = index.search(qv, fetch_k)
        scored: List[Tuple[float, float, float, int, str, Set[str]]] = []
        for rank, j in enumerate(ids[0]):
            if j < 0 or j >= len(texts):
                continue
            jj = int(j)
            body = (texts[jj] or "").strip()
            if not body:
                continue
            dense = float(sims[0][rank])
            if dense != dense:
                dense = 0.0
            toks = _rag_token_set(body)
            lex = _lexical_score(query, body, q_toks)
            combined = (1.0 - w_lex) * dense + w_lex * lex
            scored.append((combined, dense, lex, jj, body, toks))

        scored.sort(key=lambda t: -t[0])

        metrics_hits: List[Dict[str, Any]] = []
        for combined, dense, lex, jj, body, _toks in scored[: _rag_metrics_max_k()]:
            metrics_hits.append(
                {
                    "idx": jj,
                    "source": _chunk_source_at(jj),
                    "lexical_score": lex,
                    "dense_score": dense,
                    "combined_score": combined,
                }
            )
        _set_last_ranked_sources(metrics_hits)

        selected_hits: List[Dict[str, Any]] = []
        kept_tok_sets: List[Set[str]] = []
        for combined, dense, lex, jj, body, toks in scored:
            if len(selected_hits) >= int(top_k):
                break
            if kept_tok_sets and any(_rag_token_jaccard(toks, kt) >= dedup_thr for kt in kept_tok_sets):
                continue
            selected_hits.append(
                {
                    "idx": jj,
                    "body": body,
                    "source": _chunk_source_at(jj),
                    "lexical_score": lex,
                    "dense_score": dense,
                    "combined_score": combined,
                }
            )
            kept_tok_sets.append(toks)

        return _format_retrieved_evidence(
            selected_hits, max_context_chars, "RETRIEVED EVIDENCE (FAISS top-%d):\n\n"
        )
    except Exception as ex:
        print(f"RAG FAISS search failed: {ex}", flush=True)
        return None


def build_rag_context(query: str, use_rag: bool) -> Tuple[str, str, str]:
    """
    Returns (rag_block, evidence_snippet, source) where source is faiss|lexical|stub|none.
    """
    global LAST_RAG_EVIDENCE, LAST_RAG_SOURCE, LAST_RAG_HITS, LAST_RAG_RANKED_SOURCES, _RAG_MOCK_FALLBACK_WARNED
    LAST_RAG_EVIDENCE = ""
    LAST_RAG_SOURCE = "none"
    LAST_RAG_HITS = []
    LAST_RAG_RANKED_SOURCES = []
    if not use_rag:
        return "", "", "none"

    top_k = int(os.environ.get("RAG_TOP_K", "3"))
    max_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "2000"))

    def _finish(block: str, source: str) -> Tuple[str, str, str]:
        global LAST_RAG_EVIDENCE, LAST_RAG_SOURCE
        LAST_RAG_EVIDENCE = block[: min(15000, len(block))]
        LAST_RAG_SOURCE = source
        return block, LAST_RAG_EVIDENCE[:800], source

    if _rag_force_mock():
        block = _retrieve_lexical(query, top_k=top_k, max_context_chars=max_chars)
        if block:
            return _finish(block, "lexical")
        return _finish(_stub_retrieval(query, top_k=top_k), "stub")

    ensure_rag_index_env()
    _try_autobuild_rag_index_once()

    idx_path, _cp = _rag_paths()
    faiss_index_present = bool(idx_path and os.path.isfile(idx_path))

    block = _retrieve_faiss(query, top_k=top_k, max_context_chars=max_chars)
    if block:
        src = "faiss"
        method = str(LAST_RAG_DIAGNOSTIC.get("retrieval_method") or "")
        if method.startswith("hybrid"):
            src = "hybrid"
        return _finish(block, src)

    block = _retrieve_lexical(query, top_k=top_k, max_context_chars=max_chars)
    if block:
        if not _RAG_MOCK_FALLBACK_WARNED:
            _RAG_MOCK_FALLBACK_WARNED = True
            if faiss_index_present and _faiss_deps_available():
                print(
                    "RAG: FAISS index present but retrieval failed; using lexical fallback. "
                    f"Last error: {_FAISS_IMPORT_ERROR or 'see logs above'}",
                    flush=True,
                )
            else:
                print(
                    "RAG: FAISS unavailable; using corpus lexical retrieval over chunks.jsonl (once per process). "
                    "Set RAG_INDEX_DIR, pip install faiss-cpu sentence-transformers, RAG_EMBED_MODEL=all-MiniLM-L6-v2.",
                    flush=True,
                )
        return _finish(block, "lexical")

    block = _stub_retrieval(query, top_k=top_k)
    if not _RAG_MOCK_FALLBACK_WARNED:
        _RAG_MOCK_FALLBACK_WARNED = True
        print(
            "RAG: no FAISS index or chunks.jsonl corpus found; using query-only stub context. "
            "Stage PubMedQA/MedQA chunks under RAG_INDEX_DIR or /kaggle/working/rag_index.",
            flush=True,
        )
    return _finish(block, "stub")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    use_rag: bool = False,
    max_new_tokens: Optional[int] = None,
    device: str = None,
) -> Dict[str, Any]:
    """Generate one response (tokenizer + causal LM). Uses chat template when available."""
    import torch

    if max_new_tokens is None:
        raw = os.environ.get("GEN_MAX_NEW_TOKENS", "").strip()
        if raw.isdigit():
            max_new_tokens = int(raw)
        else:
            low = (prompt or "").lower()
            if "final answer letter" in low:
                if "reference:" in low:
                    max_new_tokens = 64
                else:
                    max_new_tokens = 32
            elif "reply with only the single letter" in low or "only the single letter of the best" in low:
                max_new_tokens = 32
            elif "final answer (yes, no, or maybe)" in low or (
                "exactly one word" in low and "yes, no, or maybe" in low
            ):
                max_new_tokens = 40
            else:
                max_new_tokens = 32
    max_new_tokens = max(8, min(256, int(max_new_tokens)))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rag_block, evidence_snippet, _src = build_rag_context(prompt, use_rag)
    low = (prompt or "").lower()
    is_mcq = "reply with only the single letter" in low or "single letter of the best option" in low
    is_pubmed_one_word = "exactly one word" in low and "yes, no, or maybe" in low
    if is_mcq or is_pubmed_one_word:
        # Self-contained eval prompt (reference/options or abstract already in order).
        user_content = prompt
    elif use_rag and rag_block:
        if os.environ.get("RAG_REPAIR_PROMPT", "1").strip().lower() not in ("0", "false", "no", "off"):
            user_content = f"{rag_block}\n\nClinical Answer:"
        else:
            user_content = f"Medical Context: {rag_block}\n\nQuery: {prompt}\n\nClinical Answer:"
    else:
        user_content = f"Medical Query: {prompt}\n\nClinical Answer:"

    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device

    messages = [{"role": "user", "content": user_content}]
    try:
        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("no chat_template")
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_len = len(
            tokenizer.encode(formatted, add_special_tokens=True)
        )
        if full_len > 2048:
            print(
                f"GP_BENCH: prompt truncated {full_len} -> 2048 tokens "
                "(audit M11; MCQ options or RAG evidence may be cut).",
                flush=True,
            )
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    except Exception:
        full_len = len(tokenizer.encode(user_content, add_special_tokens=True))
        if full_len > 2048:
            print(
                f"GP_BENCH: prompt truncated {full_len} -> 2048 tokens (audit M11).",
                flush=True,
            )
        inputs = tokenizer(
            user_content,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    start = time.perf_counter()
    with torch.no_grad():
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        # Pass decoding kwargs only (no generation_config, no use_model_defaults): some stacks
        # (e.g. wrapped models on Kaggle) forward unknown generate() args into model_kwargs and error.
        # kwargs are applied on top of a deepcopy of model.generation_config, overriding checkpoint
        # temperature/top_p for greedy decode (do_sample=False).
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=pad_id,
            use_cache=True,
        )
    latency = time.perf_counter() - start

    input_len = int(inputs["input_ids"].shape[1])
    response_tokens = int(outputs.shape[1]) - input_len
    try:
        from measurement_config import ENERGY_KWH_PER_TOKEN
    except ImportError:
        ENERGY_KWH_PER_TOKEN = 0.00001
    energy_kwh = response_tokens * ENERGY_KWH_PER_TOKEN
    response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    return {
        "response": response_text,
        "response_tokens": int(response_tokens),
        "latency": latency,
        "latency_seconds": latency,
        "energy_kwh": energy_kwh,
        "evidence": evidence_snippet if use_rag else "",
        "retrieved_context": LAST_RAG_EVIDENCE if use_rag else "",
        "rag_source": LAST_RAG_SOURCE if use_rag else "none",
        "rag_hits": list(LAST_RAG_HITS) if use_rag else [],
        "rag_ranked_sources": list(LAST_RAG_RANKED_SOURCES) if use_rag else [],
        "rag_diagnostic": dict(LAST_RAG_DIAGNOSTIC) if use_rag else {},
    }


def load_models(use_4bit: bool = True) -> Dict[str, Tuple[Any, Any]]:
    """
    Load SLM and LLM (see module-level MODEL_SLM / MODEL_LLM). HF token optional for public weights.
    Returns dict: {"slm": (model, tokenizer), "llm": (model, tokenizer)}.
    """
    import torch
    from transformers import AutoTokenizer

    refresh_model_ids_from_env()
    token = _get_hf_token()
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if token:
        kwargs["token"] = str(token).strip()
    elif _model_id_requires_hf_gating(MODEL_SLM) or _model_id_requires_hf_gating(MODEL_LLM):
        raise RuntimeError(
            "Hugging Face token required for the configured model id(s). "
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, or switch to open models via "
            "GP_MODEL_SLM / GP_MODEL_LLM (defaults: Gemma-2-2b-it + Llama-2-7b-chat; see real_model_runner docstring)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    fp_kwargs = {"device_map": "auto" if device == "cuda" else None}
    use_4bit_cuda = use_4bit and device == "cuda"

    if use_4bit_cuda:
        _maybe_autopip_bitsandbytes()

    if use_4bit_cuda and _bitsandbytes_import_ok():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            bnb_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        except Exception:
            bnb_kwargs = dict(fp_kwargs)
    else:
        if use_4bit_cuda and not _bitsandbytes_import_ok():
            print(
                "bitsandbytes not available or <0.46.1 after optional auto-install; loading in fp16 with "
                "device_map=auto (may OOM on one T4). Set GP_BENCH_NO_AUTO_PIP=1 to skip pip, or run: "
                "pip install -U 'bitsandbytes>=0.46.1'.",
                flush=True,
            )
        bnb_kwargs = dict(fp_kwargs)

    slm_tokenizer = AutoTokenizer.from_pretrained(MODEL_SLM, **kwargs)
    slm_model = _from_pretrained_causal_lm_safe(
        MODEL_SLM,
        model_dtype,
        bnb_kwargs,
        fp_kwargs,
        kwargs,
        use_4bit_cuda=use_4bit_cuda,
        device=device,
    )
    slm_tokenizer.pad_token = slm_tokenizer.eos_token

    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM, **kwargs)
    llm_model = _from_pretrained_causal_lm_safe(
        MODEL_LLM,
        model_dtype,
        bnb_kwargs,
        fp_kwargs,
        kwargs,
        use_4bit_cuda=use_4bit_cuda,
        device=device,
    )
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    return {
        "slm": (slm_model, slm_tokenizer),
        "llm": (llm_model, llm_tokenizer),
    }


def load_one_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any]:
    """Load a single model (one-at-a-time) to reduce peak VRAM."""
    import torch
    from transformers import AutoTokenizer

    if model_key not in ("slm", "llm"):
        raise ValueError(f"model_key must be 'slm' or 'llm', got: {model_key}")

    refresh_model_ids_from_env()
    token = _get_hf_token()
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if token:
        kwargs["token"] = str(token).strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    fp_kwargs = {"device_map": "auto" if device == "cuda" else None}
    use_4bit_cuda = use_4bit and device == "cuda"

    if use_4bit_cuda:
        _maybe_autopip_bitsandbytes()

    if use_4bit_cuda and _bitsandbytes_import_ok():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            bnb_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        except Exception:
            bnb_kwargs = dict(fp_kwargs)
    else:
        if use_4bit_cuda and not _bitsandbytes_import_ok():
            print(
                "bitsandbytes not available or <0.46.1 after optional auto-install; loading in fp16 with "
                "device_map=auto (may OOM on one T4). Set GP_BENCH_NO_AUTO_PIP=1 to skip pip, or run: "
                "pip install -U 'bitsandbytes>=0.46.1'.",
                flush=True,
            )
        bnb_kwargs = dict(fp_kwargs)

    if model_key == "slm":
        model_name = MODEL_SLM
    else:
        model_name = MODEL_LLM

    if not token and _model_id_requires_hf_gating(model_name):
        raise RuntimeError(
            "Hugging Face token required for this model id. Set HF_TOKEN or use GP_MODEL_SLM / "
            "GP_MODEL_LLM with a public checkpoint (defaults: Gemma-2-2b-it + Llama-2-7b-chat)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = _from_pretrained_causal_lm_safe(
        model_name,
        model_dtype,
        bnb_kwargs,
        fp_kwargs,
        kwargs,
        use_4bit_cuda=use_4bit_cuda,
        device=device,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_single(query: str, model_key: str, use_rag: bool, models_dict: Dict) -> Dict[str, Any]:
    """One inference: query + model_key (slm/llm) + use_rag."""
    model, tokenizer = models_dict[model_key]
    out = generate_response(model, tokenizer, query, use_rag=use_rag)
    out["model"] = model_key
    out["rag"] = use_rag
    out["query"] = query
    return out
