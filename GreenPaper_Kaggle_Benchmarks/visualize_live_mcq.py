"""
Live MedQA-style MCQ viewer (default n=150): shows each vignette, options, gold letter,
model raw output, and parsed letter as inference runs.

Uses the same data loading and letter parsing as ``eval_benchmarks.py`` (imported helpers).
Prompt ends with: reply with only the answer (single letter A–D).

Examples
--------
  pip install gradio datasets
  # HF Router (no local GPU):
  set HF_TOKEN=...
  python visualize_live_mcq.py --hf_router --n 150 --seed 42

  # Local GPU (same stack as eval_benchmarks):
  python visualize_live_mcq.py --n 150 --seed 42

  # Terminal only (no Gradio):
  python visualize_live_mcq.py --mock --n 5 --no_gradio

  # One config only (default SLM_NoRAG):
  python visualize_live_mcq.py --config LLM_NoRAG --hf_router --n 150

**Kaggle / Jupyter:** Put ``eval_benchmarks.py`` and ``visualize_live_mcq.py`` in the same folder
(typically ``/kaggle/working``) or add that folder to ``sys.path`` before importing. This file
auto-prepends sensible search paths so ``import eval_benchmarks`` works from notebooks and
``%run`` cells.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


def _bootstrap_eval_benchmarks_path() -> None:
    """
    Ensure the directory that contains ``eval_benchmarks.py`` is on ``sys.path``.

    Fixes ``ModuleNotFoundError`` when:
    - Running from a Kaggle/Jupyter cwd that is not the script folder
    - Pasting code into a notebook (no ``__file__``): still checks cwd and ``/kaggle/working``
    """
    candidates: List[str] = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    candidates.append(os.getcwd())
    if os.path.isdir("/kaggle/working"):
        candidates.append("/kaggle/working")
    kin = "/kaggle/input"
    if os.path.isdir(kin):
        try:
            for name in sorted(os.listdir(kin)):
                root = os.path.join(kin, name)
                if not os.path.isdir(root):
                    continue
                candidates.append(root)
                try:
                    for sub in sorted(os.listdir(root)):
                        sp = os.path.join(root, sub)
                        if os.path.isdir(sp):
                            candidates.append(sp)
                except OSError:
                    pass
        except OSError:
            pass
    code_dir = (os.environ.get("CODE_DIR") or os.environ.get("GREEN_PAPER_CODE") or "").strip()
    if code_dir:
        candidates.insert(0, code_dir)

    seen: set[str] = set()
    for d in candidates:
        d = os.path.abspath(d)
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        if os.path.isfile(os.path.join(d, "eval_benchmarks.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return
    # Prefer script dir / cwd on path even if eval_benchmarks not found (clearer follow-up error).
    for d in candidates:
        d = os.path.abspath(d)
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
            break


_bootstrap_eval_benchmarks_path()

# Reuse benchmark helpers from the same package directory.
try:
    from eval_benchmarks import (
        CONFIGS,
        _drop_runner_caches,
        _ensure_hf_token,
        _extract_mcq_letter,
        _hf_router_max_tokens,
        _load_items,
        _normalize_mcq_gold,
        _resolve_hf_router_models,
        _run_out,
        _setup_runner_path,
    )
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "eval_benchmarks":
        raise ModuleNotFoundError(
            "Could not import eval_benchmarks. On Kaggle, place eval_benchmarks.py in the same "
            "directory as this file (e.g. /kaggle/working), or set environment variable CODE_DIR to "
            "that folder, then re-run. From a notebook you can also run: "
            "import sys; sys.path.insert(0, '/kaggle/working')"
        ) from e
    raise


def _mcq_prompt(question: str, choices: List[str]) -> str:
    lines = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{question.strip()}\n\n"
        f"Options:\n{lines}\n\n"
        "Reply with only the answer (single letter A, B, C, or D)."
    )


def _build_run_fn(
    *,
    mock: bool,
    hf_router: bool,
    hf_router_base_url: str,
    hf_router_model_slm: str,
    hf_router_model_llm: str,
    use_4bit: bool,
) -> Callable[[str, str, bool], Any]:
    if mock and hf_router:
        raise RuntimeError("--mock and --hf_router cannot be used together.")

    if mock:

        def run_fn(model_key: str, question: str, use_rag: bool) -> str:
            return "A"

    elif hf_router:
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
            raise RuntimeError("HF_TOKEN required for --hf_router.")

        mid_slm, mid_llm = _resolve_hf_router_models(hf_router_model_slm, hf_router_model_llm)
        client = OpenAI(api_key=str(token).strip(), base_url=base)

        def run_fn(model_key: str, question: str, use_rag: bool) -> Dict[str, Any]:
            rag_block, _ev_snip, src = _rmr.build_rag_context(question, use_rag)
            if use_rag and rag_block:
                user_content = f"Medical Context: {rag_block}\n\nQuery: {question}\n\nClinical Answer:"
            else:
                user_content = f"Medical Query: {question}\n\nClinical Answer:"
            model_id = mid_slm if model_key == "slm" else mid_llm
            comp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=_hf_router_max_tokens(),
                temperature=0,
            )
            text = (comp.choices[0].message.content or "").strip()
            return {
                "response": text,
                "retrieved_context": _rmr.LAST_RAG_EVIDENCE if use_rag else "",
                "rag_source": src if use_rag else "none",
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
            from real_model_runner import load_models  # type: ignore

            def load_one_model(model_key: str, use_4bit: bool = True) -> Tuple[Any, Any]:
                models_all = load_models(use_4bit=use_4bit)
                return models_all[model_key]

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
            assert model_key in ("slm", "llm")
            m, tok = load_one_model(model_key, use_4bit=use_4bit)
            models[model_key] = (m, tok)
            _active_model_key = model_key

        def run_fn(model_key: str, question: str, use_rag: bool) -> Any:
            _ensure_model_loaded(model_key)
            return run_single(question, model_key=model_key, use_rag=use_rag, models_dict=models)

    return run_fn


def _config_by_name(name: str) -> Tuple[str, str, bool]:
    for cfg_name, model_key, use_rag in CONFIGS:
        if cfg_name == name:
            return cfg_name, model_key, use_rag
    raise ValueError(f"Unknown --config {name!r}. Choose one of: {[c[0] for c in CONFIGS]}")


def _format_item_block(
    idx: int,
    total: int,
    item: Dict[str, Any],
    raw: str,
    pred: str,
    gold: str,
    ok: bool,
) -> str:
    qid = str(item.get("id", ""))
    correct = "Yes" if ok else "No"
    body = "\n".join(
        f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(item.get("choices") or []) if c is not None
    )
    return (
        f"### Item {idx + 1} / {total} (id `{qid}`)\n\n"
        f"**Stem**\n\n{item.get('question', '').strip()}\n\n"
        f"**Options**\n\n{body}\n\n"
        f"**Gold** `{gold}` &nbsp;|&nbsp; **Parsed** `{pred or '?'}` &nbsp;|&nbsp; **Match** {correct}\n\n"
        f"**Model reply (raw)**\n\n```\n{raw.strip() or '(empty)'}\n```\n\n"
        "---\n\n"
    )


def iter_mcq_steps(
    benchmark: str,
    data_path: str,
    n: int,
    seed: Optional[int],
    cfg_name: str,
    run_fn: Callable[[str, str, bool], Any],
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """Yields (markdown_log, progress_line, last_row_dict) per item."""
    items = _load_items(benchmark, data_path, n, seed)
    if not items:
        yield ("No items loaded.", "0/0", {})
        return

    _, model_key, use_rag = _config_by_name(cfg_name)
    log_parts: List[str] = []
    hits = 0

    for i, item in enumerate(items):
        choices = [c for c in item["choices"] if c is not None]
        nch = len(choices)
        gold = _normalize_mcq_gold(str(item["answer"]), nch)
        prompt = _mcq_prompt(str(item.get("question", "")), choices)
        out = run_fn(model_key, prompt, use_rag)
        raw, rctx, rsrc, _rhits, _rranked, _rdiag = _run_out(out)
        pred = _extract_mcq_letter(raw, nch)
        ok = pred == gold and gold != ""
        if ok:
            hits += 1

        log_parts.append(_format_item_block(i, len(items), item, raw, pred, gold, ok))
        progress = f"{i + 1}/{len(items)} | running accuracy {hits}/{i + 1} ({100.0 * hits / (i + 1):.1f}%)"
        row = {
            "index": i,
            "question_id": str(item.get("id", "")),
            "gold": gold,
            "pred": pred,
            "raw": raw,
            "correct": ok,
            "rag": use_rag,
            "retrieved_context": rctx if use_rag else "",
            "rag_source": rsrc if use_rag else "",
        }
        yield ("".join(log_parts), progress, row)


def _run_terminal(
    benchmark: str,
    data_path: str,
    n: int,
    seed: Optional[int],
    cfg_name: str,
    run_fn: Callable[[str, str, bool], Any],
    out_jsonl: str,
) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    rows: List[Dict[str, Any]] = []
    last_md = ""
    iterator = iter_mcq_steps(benchmark, data_path, n, seed, cfg_name, run_fn)
    if tqdm:
        iterator = tqdm(iterator, desc="MCQ", unit="q")

    for md, prog, row in iterator:
        last_md = md
        rows.append(row)
        print(prog, flush=True)
        if row:
            print(f"  id={row.get('question_id')} gold={row.get('gold')} pred={row.get('pred')} ok={row.get('correct')}", flush=True)

    if out_jsonl:
        ap = os.path.abspath(out_jsonl)
        parent = os.path.dirname(ap)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_jsonl}", flush=True)

    # Print short tail so terminal users see the last case without scrolling the full md.
    if last_md:
        tail = last_md[-1200:] if len(last_md) > 1200 else last_md
        print("\n--- last log tail ---\n", tail, flush=True)


def _run_gradio(
    benchmark: str,
    data_path: str,
    n: int,
    seed: Optional[int],
    cfg_name: str,
    run_fn: Callable[[str, str, bool], Any],
    server_port: int,
) -> None:
    import gradio as gr

    def go():
        for md, prog, _row in iter_mcq_steps(benchmark, data_path, n, seed, cfg_name, run_fn):
            yield md, prog

    with gr.Blocks(title="Live MCQ") as demo:
        gr.Markdown(
            f"**Benchmark:** `{benchmark}` &nbsp;|&nbsp; **n**={n} &nbsp;|&nbsp; **config** `{cfg_name}`"
        )
        log = gr.Markdown(value="_Click Run to start._")
        prog = gr.Textbox(label="Progress", interactive=False)
        run_btn = gr.Button("Run")
        run_btn.click(fn=go, inputs=[], outputs=[log, prog])

    demo.launch(server_port=server_port, share=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Live MCQ visualization (default MedQA n=150).")
    p.add_argument("--benchmark", default="medqa", choices=["medqa", "medmcqa", "mmlu_med"])
    p.add_argument("--data_path", default="", help="Optional JSON/JSONL for custom rows (see eval_benchmarks).")
    p.add_argument("--n", type=int, default=150, help="Max items (default 150).")
    p.add_argument("--seed", type=int, default=42, help="Subset seed (same semantics as eval --seed).")
    p.add_argument(
        "--config",
        default="SLM_NoRAG",
        choices=[c[0] for c in CONFIGS],
        help="Single eval config to run (avoids 4× API/GPU cost).",
    )
    p.add_argument("--mock", action="store_true")
    p.add_argument("--hf_router", action="store_true")
    p.add_argument("--hf_router_base_url", default="")
    p.add_argument("--hf_router_model_slm", default="")
    p.add_argument("--hf_router_model_llm", default="")
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--no_gradio", action="store_true", help="Terminal + tqdm only.")
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--out_jsonl", default="", help="If set, append one JSON object per item (terminal mode).")
    args = p.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    run_fn = _build_run_fn(
        mock=args.mock,
        hf_router=args.hf_router,
        hf_router_base_url=args.hf_router_base_url,
        hf_router_model_slm=args.hf_router_model_slm,
        hf_router_model_llm=args.hf_router_model_llm,
        use_4bit=not args.no_4bit,
    )

    if args.no_gradio:
        _run_terminal(
            args.benchmark,
            args.data_path.strip(),
            args.n,
            args.seed,
            args.config,
            run_fn,
            args.out_jsonl.strip(),
        )
        return

    try:
        import gradio  # noqa: F401
    except ImportError as e:
        raise SystemExit("pip install gradio (or pass --no_gradio for terminal-only).") from e

    _run_gradio(
        args.benchmark,
        args.data_path.strip(),
        args.n,
        args.seed,
        args.config,
        run_fn,
        args.server_port,
    )


if __name__ == "__main__":
    main()
