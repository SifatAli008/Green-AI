from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path


def main() -> None:
    eval_path = Path(r"d:\Green Paper\code\eval_benchmarks.py")
    text = eval_path.read_text(encoding="utf-8")

    marker = "_BUNDLED_PY_ZIP_B64 = \"\"\""
    start = text.find(marker)
    if start < 0:
        raise SystemExit("Could not find marker")
    start += len(marker)
    end = text.find("\"\"\"", start)
    if end < 0:
        raise SystemExit("Could not find end delimiter")

    b64 = text[start:end].strip()
    raw = base64.b64decode(b64)
    zf = zipfile.ZipFile(io.BytesIO(raw))
    runner = zf.read("real_model_runner.py").decode("utf-8")
    print("load_one_model" if "load_one_model" in runner else "MISSING", flush=True)


if __name__ == "__main__":
    main()

