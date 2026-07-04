"""Build paragraph-level PubMedQA chunks (see build_pubmedqa_chunks.py)."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from build_pubmedqa_chunks import build_chunks  # noqa: E402
import json  # noqa: E402

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    chunks = build_chunks(split="train", min_chars=40, max_items=500, seed=42)
    out_jsonl = os.path.join(WORKDIR, "medical_chunks.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} paragraph chunks -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
