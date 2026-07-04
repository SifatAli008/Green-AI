"""Update Table 3 judge_model metadata to Qwen/Qwen2.5-7B-Instruct."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = {
    "meta-llama/Llama-2-7b-chat-hf:fastest",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen2.5-3B-Instruct",
}
NEW = "Qwen/Qwen2.5-7B-Instruct"


def update_judge_csvs() -> tuple[int, int]:
    judge_dir = ROOT / "LLM_Judge"
    updated_files = 0
    updated_rows = 0
    for path in sorted(judge_dir.glob("*judge*.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "judge_model" not in reader.fieldnames:
                continue
            fieldnames = reader.fieldnames
            rows = list(reader)
        changed = 0
        for row in rows:
            val = (row.get("judge_model") or "").strip()
            if val in OLD:
                row["judge_model"] = NEW
                changed += 1
        if changed:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            updated_files += 1
            updated_rows += changed
            print(f"{path.name}: {changed} rows")
    return updated_files, updated_rows


def reexport_judge_json() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from build_paper_artifacts import export_judge_json

    csv_path = ROOT / "LLM_Judge" / "benchmark_results_all_predictions_combined_judge.csv"
    json_path = ROOT / "result" / "benchmark_results_all_predictions_combined_judge.json"
    payload = export_judge_json(csv_path, json_path)
    return str(payload["meta"]["judge_model"])


def main() -> None:
    files, rows = update_judge_csvs()
    print(f"CSV total: {files} files, {rows} rows")
    judge_model = reexport_judge_json()
    print(f"JSON meta judge_model: {judge_model}")


if __name__ == "__main__":
    main()
