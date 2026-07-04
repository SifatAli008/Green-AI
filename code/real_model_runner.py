"""
Real model inference for full evaluation. Uses Hugging Face token from environment only.
Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running; never hardcode the token.
Models: SLM = google/gemma-2b, LLM = meta-llama/Llama-2-7b-hf.

RAG (real FAISS): set RAG_INDEX_DIR (folder with index.faiss + chunks.jsonl) or
RAG_FAISS_INDEX + RAG_CHUNKS_JSONL. Embeddings must match RAG_EMBED_MODEL (default
sentence-transformers/all-MiniLM-L6-v2), L2-normalized IndexFlatIP. If unset or load fails,
the runner looks for index.faiss + chunks.jsonl under /kaggle/working/rag_index, ./rag_index,
then /kaggle/input/*/rag_index and /kaggle/input/* (first match wins). Otherwise falls back to
mock retrieval. See build_rag_index.py to build the index.

Ablations (env, or set by eval_benchmarks.py --rag_* flags):
  RAG_TOP_K, RAG_CONTEXT_MAX_CHARS, RAG_FORCE_MOCK=1, RAG_EMBED_MODEL
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Filled when use_rag=True so eval_benchmarks can log retrieved_context (single-threaded runs).
LAST_RAG_EVIDENCE: str = ""
LAST_RAG_SOURCE: str = ""  # "faiss" | "mock" | "none"
_RAG_MOCK_FALLBACK_WARNED: bool = False


def _get_hf_token() -> Optional[str]:
    """Read Hugging Face token from environment. Never log or store it."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def mock_retrieval(query: str, top_k: int = 3) -> str:
    """Simulate RAG retrieval when no index is configured or RAG_FORCE_MOCK=1."""
    return f"""
RETRIEVED EVIDENCE ({top_k} chunks):
Chunk 1: [PubMed 2025]: Current guidelines recommend metformin as first-line...
Chunk 2: [FDA Guideline]: SGLT2 inhibitors approved for cardio-renal protection...
Chunk 3: [NICE 2026]: Target HbA1c <7.0% with individualized therapy...
"""


def _rag_force_mock() -> bool:
    return os.environ.get("RAG_FORCE_MOCK", "").strip() in ("1", "true", "yes", "on")


def _rag_pair_in_dir(d: str) -> Tuple[str, str]:
    return os.path.join(d, "index.faiss"), os.path.join(d, "chunks.jsonl")


def _iter_autodiscover_rag_dirs() -> List[str]:
    """Kaggle-friendly search order when RAG_* env is missing or paths are invalid."""
    seen: set[str] = set()
    out: List[str] = []

    def push(p: str) -> None:
        ap = os.path.abspath(os.path.normpath(p))
        if ap in seen or not os.path.isdir(ap):
            return
        seen.add(ap)
        out.append(ap)

    if os.path.isdir("/kaggle"):
        push("/kaggle/working/rag_index")
    push(os.path.join(os.getcwd(), "rag_index"))
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        for name in sorted(os.listdir(kin)):
            root = os.path.join(kin, name)
            if os.path.isdir(root):
                push(os.path.join(root, "rag_index"))
                push(root)
    return out


def _rag_paths() -> Tuple[str, str]:
    """Return (faiss_path, chunks_jsonl_path) or ("", "")."""
    idx = os.environ.get("RAG_FAISS_INDEX", "").strip()
    chunks = os.environ.get("RAG_CHUNKS_JSONL", "").strip()
    d = os.environ.get("RAG_INDEX_DIR", "").strip()
    if d and (not idx or not chunks):
        idx, chunks = _rag_pair_in_dir(d)
    if idx and chunks and os.path.isfile(idx) and os.path.isfile(chunks):
        return idx, chunks
    for cand in _iter_autodiscover_rag_dirs():
        ip, cp = _rag_pair_in_dir(cand)
        if os.path.isfile(ip) and os.path.isfile(cp):
            print(
                f"RAG: auto-discovered FAISS index in {cand} "
                "(set RAG_INDEX_DIR or --rag_index_dir to pin a path).",
                flush=True,
            )
            return ip, cp
    return idx, chunks


_rag_singleton: Dict[str, Any] = {}


def _clear_rag_cache() -> None:
    global _RAG_MOCK_FALLBACK_WARNED
    _rag_singleton.clear()
    _RAG_MOCK_FALLBACK_WARNED = False


def _load_chunk_texts(path: str) -> List[str]:
    import json

    texts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            if isinstance(obj, dict):
                t = obj.get("text") or obj.get("chunk") or obj.get("content") or ""
                texts.append(str(t))
            else:
                texts.append(str(obj))
    return texts


def _retrieve_faiss(query: str, top_k: int, max_context_chars: int) -> Optional[str]:
    """
    Return formatted evidence string, or None if unavailable / error.
    """
    global _rag_singleton
    idx_path, chunk_path = _rag_paths()
    if not idx_path or not chunk_path or not os.path.isfile(idx_path) or not os.path.isfile(chunk_path):
        return None

    try:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    cache_key = f"{idx_path}|{chunk_path}|{os.environ.get('RAG_EMBED_MODEL', '')}"
    if _rag_singleton.get("key") != cache_key:
        _rag_singleton.clear()
        _rag_singleton["key"] = cache_key
        embed_name = (
            os.environ.get("RAG_EMBED_MODEL", "").strip()
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        st = SentenceTransformer(embed_name)
        index = faiss.read_index(idx_path)
        texts = _load_chunk_texts(chunk_path)
        if index.ntotal != len(texts):
            print(
                f"RAG warning: FAISS ntotal={index.ntotal} != len(chunks)={len(texts)}; "
                "truncating/padding to match (check build_rag_index.py).",
                flush=True,
            )
            if len(texts) > index.ntotal:
                texts = texts[: index.ntotal]
            else:
                texts.extend([""] * (index.ntotal - len(texts)))
        _rag_singleton["index"] = index
        _rag_singleton["texts"] = texts
        _rag_singleton["st"] = st
        print(
            f"RAG: loaded FAISS index ({index.ntotal} vectors) + chunks from {chunk_path}",
            flush=True,
        )

    index = _rag_singleton["index"]
    texts: List[str] = _rag_singleton["texts"]
    st = _rag_singleton["st"]

    try:
        import numpy as np
        import faiss  # type: ignore

        try:
            qv = st.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            qv = st.encode([query], convert_to_numpy=True)
        if qv.dtype != np.float32:
            qv = qv.astype(np.float32)
        faiss.normalize_L2(qv)
        k = min(int(top_k), int(index.ntotal), max(1, index.ntotal))
        sims, ids = index.search(qv, k)
        lines: List[str] = []
        used = 0
        for rank, j in enumerate(ids[0]):
            if j < 0 or j >= len(texts):
                continue
            body = (texts[j] or "").strip()
            if not body:
                continue
            chunk_line = f"Chunk {rank + 1} (idx={int(j)}):\n{body}"
            if used + len(chunk_line) + 2 > max_context_chars:
                remain = max_context_chars - used - 50
                if remain > 80:
                    chunk_line = f"Chunk {rank + 1} (idx={int(j)}):\n{body[:remain]}..."
                else:
                    break
            lines.append(chunk_line)
            used += len(chunk_line) + 2
        if not lines:
            return None
        return "RETRIEVED EVIDENCE (FAISS top-%d):\n\n" % len(lines) + "\n\n".join(lines)
    except Exception as ex:
        print(f"RAG FAISS search failed: {ex}", flush=True)
        return None


def build_rag_context(query: str, use_rag: bool) -> Tuple[str, str, str]:
    """
    Returns (rag_block, evidence_snippet, source) where source is faiss|mock|none.
    """
    global LAST_RAG_EVIDENCE, LAST_RAG_SOURCE, _RAG_MOCK_FALLBACK_WARNED
    LAST_RAG_EVIDENCE = ""
    LAST_RAG_SOURCE = "none"
    if not use_rag:
        return "", "", "none"

    top_k = int(os.environ.get("RAG_TOP_K", "5"))
    max_chars = int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "6000"))

    if _rag_force_mock():
        block = mock_retrieval(query, top_k=top_k)
        LAST_RAG_EVIDENCE = block[: min(15000, len(block))]
        LAST_RAG_SOURCE = "mock"
        return block, LAST_RAG_EVIDENCE[:800], "mock"

    block = _retrieve_faiss(query, top_k=top_k, max_context_chars=max_chars)
    if block:
        LAST_RAG_EVIDENCE = block[: min(15000, len(block))]
        LAST_RAG_SOURCE = "faiss"
        return block, LAST_RAG_EVIDENCE[:800], "faiss"

    block = mock_retrieval(query, top_k=top_k)
    LAST_RAG_EVIDENCE = block[: min(15000, len(block))]
    LAST_RAG_SOURCE = "mock"
    if not _RAG_MOCK_FALLBACK_WARNED:
        _RAG_MOCK_FALLBACK_WARNED = True
        print(
            "RAG: FAISS not configured or failed; using mock retrieval (once per process). "
            "Build index with build_rag_index.py, place under /kaggle/working/rag_index or set "
            "RAG_INDEX_DIR / --rag_index_dir, or RAG_FAISS_INDEX+RAG_CHUNKS_JSONL. "
            "On one T4 without 4-bit (no bitsandbytes), consider --max_items 150 (or lower) to cut VRAM and runtime.",
            flush=True,
        )
    return block, LAST_RAG_EVIDENCE[:800], "mock"


def _generation_kwargs_for_prompt(prompt: str) -> Dict[str, Any]:
    """
    Task-aware decoding: short greedy outputs for MCQ / PubMedQA labels (faster, same eval intent).
    Override with GEN_MAX_NEW_TOKENS / GEN_MIN_NEW_TOKENS / GEN_DO_SAMPLE=1.
    """
    low = (prompt or "").lower()
    if os.environ.get("GEN_DO_SAMPLE", "").strip() in ("1", "true", "yes", "on"):
        return {
            "max_new_tokens": int(os.environ.get("GEN_MAX_NEW_TOKENS", "128")),
            "min_new_tokens": int(os.environ.get("GEN_MIN_NEW_TOKENS", "0")),
            "temperature": float(os.environ.get("GEN_TEMPERATURE", "0.7")),
            "top_p": float(os.environ.get("GEN_TOP_P", "0.9")),
            "do_sample": True,
            "repetition_penalty": float(os.environ.get("GEN_REPETITION_PENALTY", "1.1")),
        }
    if "reply with only the single letter" in low or "only the single letter of the best" in low:
        return {
            "max_new_tokens": int(os.environ.get("GEN_MAX_NEW_TOKENS", "16")),
            "min_new_tokens": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
    if "exactly one word" in low or "yes, no, or maybe only" in low:
        return {
            "max_new_tokens": int(os.environ.get("GEN_MAX_NEW_TOKENS", "24")),
            "min_new_tokens": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
    return {
        "max_new_tokens": int(os.environ.get("GEN_MAX_NEW_TOKENS", "128")),
        "min_new_tokens": int(os.environ.get("GEN_MIN_NEW_TOKENS", "0")),
        "temperature": float(os.environ.get("GEN_TEMPERATURE", "0.3")),
        "top_p": float(os.environ.get("GEN_TOP_P", "0.9")),
        "do_sample": False,
        "repetition_penalty": float(os.environ.get("GEN_REPETITION_PENALTY", "1.05")),
    }


def prewarm_rag_index() -> bool:
    """Load FAISS + embedder once before the benchmark loop (no-op if RAG unset)."""
    if _rag_force_mock():
        return False
    block, _, src = build_rag_context("warmup biomedical retrieval query", True)
    return bool(block) and src == "faiss"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    use_rag: bool = False,
    max_new_tokens: int = 128,
    min_new_tokens: int = 64,
    device: str = None,
) -> Dict[str, Any]:
    """Generate one response (tokenizer + causal LM)."""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rag_block, evidence_snippet, _src = build_rag_context(prompt, use_rag)
    if use_rag and rag_block:
        full_prompt = f"Medical Context: {rag_block}\n\nQuery: {prompt}\n\nClinical Answer:"
    else:
        full_prompt = f"Medical Query: {prompt}\n\nClinical Answer:"

    from transformers import GenerationConfig

    gkw = _generation_kwargs_for_prompt(prompt)
    gen_config = GenerationConfig(
        max_new_tokens=gkw["max_new_tokens"],
        min_new_tokens=gkw["min_new_tokens"],
        temperature=gkw["temperature"],
        top_p=gkw["top_p"],
        do_sample=gkw["do_sample"],
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=gkw["repetition_penalty"],
    )

    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    start = time.perf_counter()
    with torch.no_grad():
        inputs = tokenizer(
            full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        outputs = model.generate(**inputs, generation_config=gen_config)
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
    }


def load_models(use_4bit: bool = True) -> Dict[str, Tuple[Any, Any]]:
    """
    Load SLM (Gemma-2B) and LLM (Llama-2-7B). Uses HF token from env if set.
    Returns dict: {"slm": (model, tokenizer), "llm": (model, tokenizer)}.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not set. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment "
            "to load gated models (e.g. meta-llama/Llama-2-7b-hf)."
        )
    kwargs = {"token": token, "trust_remote_code": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    def _bitsandbytes_usable() -> bool:
        try:
            from transformers.utils.import_utils import is_bitsandbytes_available

            return bool(is_bitsandbytes_available())
        except Exception:
            try:
                import bitsandbytes  # noqa: F401

                return True
            except ImportError:
                return False

    if use_4bit and device == "cuda" and _bitsandbytes_usable():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            bnb_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        except Exception:
            bnb_kwargs = {"device_map": "auto" if device == "cuda" else None}
    else:
        if use_4bit and device == "cuda" and not _bitsandbytes_usable():
            print(
                "bitsandbytes not available; loading models in fp16 with device_map=auto (may OOM on one T4).",
                flush=True,
            )
        bnb_kwargs = {"device_map": "auto" if device == "cuda" else None}

    slm_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", **kwargs)
    slm_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b",
        dtype=torch_dtype,
        **bnb_kwargs,
        **kwargs,
    )
    slm_tokenizer.pad_token = slm_tokenizer.eos_token

    llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", **kwargs)
    llm_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        dtype=torch_dtype,
        **bnb_kwargs,
        **kwargs,
    )
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    return {
        "slm": (slm_model, slm_tokenizer),
        "llm": (llm_model, llm_tokenizer),
    }


def load_one_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any]:
    """Load a single model (one-at-a-time) to reduce peak VRAM."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if model_key not in ("slm", "llm"):
        raise ValueError(f"model_key must be 'slm' or 'llm', got: {model_key}")

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not set. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment "
            "to load gated models (e.g. meta-llama/Llama-2-7b-hf)."
        )

    kwargs = {"token": token, "trust_remote_code": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    def _bitsandbytes_usable() -> bool:
        try:
            from transformers.utils.import_utils import is_bitsandbytes_available

            return bool(is_bitsandbytes_available())
        except Exception:
            try:
                import bitsandbytes  # noqa: F401

                return True
            except ImportError:
                return False

    if use_4bit and device == "cuda" and _bitsandbytes_usable():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            bnb_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        except Exception:
            bnb_kwargs = {"device_map": "auto" if device == "cuda" else None}
    else:
        if use_4bit and device == "cuda" and not _bitsandbytes_usable():
            print(
                "bitsandbytes not available; loading models in fp16 with device_map=auto (may OOM on one T4).",
                flush=True,
            )
        bnb_kwargs = {"device_map": "auto" if device == "cuda" else None}

    if model_key == "slm":
        model_name = "google/gemma-2b"
    else:
        model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        **bnb_kwargs,
        **kwargs,
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
