"""Zip measurement_config, eval_quality_metrics, real_model_runner → embed in eval_benchmarks.py."""
from __future__ import annotations

import base64
import io
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["measurement_config.py", "eval_quality_metrics.py", "real_model_runner.py"]
EVAL = ROOT / "eval_benchmarks.py"


def main() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.writestr(name, (ROOT / name).read_bytes())
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    text = EVAL.read_text(encoding="utf-8")
    m = re.search(r'_BUNDLED_PY_ZIP_B64 = """[\s\S]*?"""', text)
    if not m:
        raise SystemExit("_BUNDLED_PY_ZIP_B64 block not found")
    new_text = text[: m.start()] + '_BUNDLED_PY_ZIP_B64 = """' + b64 + '"""' + text[m.end() :]
    EVAL.write_text(new_text, encoding="utf-8")
    print(f"Embedded bundle: {len(b64)} base64 chars")


if __name__ == "__main__":
    main()
