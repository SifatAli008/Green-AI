"""One-off: refresh _BUNDLED_PY_ZIP_B64 in eval_benchmarks.py from helper sources."""
import base64
import io
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = (
    "measurement_config.py",
    "eval_quality_metrics.py",
    "rag_retrieval.py",
    "real_model_runner.py",
)

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in FILES:
        zf.write(ROOT / name, arcname=name)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")
text = (ROOT / "eval_benchmarks.py").read_text(encoding="utf-8")
pat = r'(_BUNDLED_PY_ZIP_B64 = )"""[^"]*"""'
new = f'_BUNDLED_PY_ZIP_B64 = """{b64}"""'
out, n = re.subn(pat, new, text, count=1)
if n != 1:
    raise SystemExit(f"replace failed: {n}")
(ROOT / "eval_benchmarks.py").write_text(out, encoding="utf-8")
print(f"OK: {len(b64)} b64 chars, {len(buf.getvalue())} zip bytes")
