"""Build ../GreenPaper_Kaggle_Benchmarks.zip for drag-and-drop (Kaggle dataset or local backup)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
OUT = ROOT / "GreenPaper_Kaggle_Benchmarks.zip"

FILES = [
    "eval_benchmarks.py",
    "real_model_runner.py",
    "eval_quality_metrics.py",
    "measurement_config.py",
    "requirements_kaggle.txt",
]

README = """Green Paper — Kaggle benchmark bundle
=====================================

Always copy ``eval_benchmarks.py`` from ``code/`` in this repo (or rebuild this zip).
Do not use an old unzipped ``GreenPaper_Kaggle_Benchmarks/`` folder from a previous download.

Single-file option
------------------
``eval_benchmarks.py`` embeds the three helper modules; on Kaggle it unpacks them into
``/kaggle/working`` automatically. You may upload **only** that file—this zip is optional.

Contents
--------
- eval_benchmarks.py      Main script (self-extracts helpers if needed)
- real_model_runner.py    Gemma + Llama loaders (also embedded in eval_benchmarks.py)
- eval_quality_metrics.py Metrics helpers
- measurement_config.py   Energy constants
- requirements_kaggle.txt Extra pip deps reference

Kaggle (drag-and-drop)
----------------------
1. Upload this ZIP as a Kaggle Dataset (or unzip into /kaggle/working in a notebook),
   **or** upload only ``eval_benchmarks.py``.
2. In a code cell before running:
     !pip install -q datasets rouge-score
3. Set Secret HF_TOKEN (or env) for Llama-2 access.
4. Run:
     !python /kaggle/working/eval_benchmarks.py --mock   # smoke test
     !python /kaggle/working/eval_benchmarks.py         # real models

Optional 4-bit (less VRAM): !pip install -q 'bitsandbytes>=0.46.1'
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BUNDLE_README.txt", README)
        for name in FILES:
            path = CODE / name
            if not path.is_file():
                raise SystemExit(f"Missing file: {path}")
            zf.write(path, arcname=name)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
