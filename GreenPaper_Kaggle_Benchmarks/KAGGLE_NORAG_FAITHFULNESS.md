# Kaggle — NoRAG faithfulness (Table 4 gap-fill)

RAG faithfulness is **already complete** (18,180 rows). This run scores the remaining **18,180 NoRAG** rows so Table 4 can compare grounding fairly:

| Configuration | Context judged against |
|---------------|------------------------|
| RAG | Retrieved evidence (done) |
| NoRAG | PubMedQA abstract, or MCQ question + choices |

**GPU:** T4 is enough (no FAISS / RAG indexes required for NoRAG).

---

## 1. Attach datasets

| Dataset | Purpose |
|---------|---------|
| `fatinshadab/benchmark-results-all-predictions-combined` | Predictions CSV (36,360 rows) |
| *(optional)* your uploaded RAG faithfulness rerun | Prior RAG batches for merge metadata |

---

## 2. Notebook cells

### Cell 0 — pip (once per session)

```python
!pip install -q 'transformers>=4.43' accelerate bitsandbytes
```

Restart kernel after pip.

### Cell 1 — paste entire `run_faithfulness_eval.py` and Run

Set `GP_FAITH_AUTO=0` so it does not auto-start the legacy RAG-only pass:

```python
import os
os.environ["GP_FAITH_AUTO"] = "0"
```

### Cell 2 — secrets + NoRAG env

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["GP_FAITH_AUTO"] = "0"
os.environ["GP_FAITH_BATCH_SIZE"] = "500"
```

### Cell 3 — run NoRAG batches (re-run until complete)

```python
# Optional: path to attached RAG faithfulness folder (for merge / prior key scan)
PRIOR_RAG = "/kaggle/input/YOUR_RAG_FAITH_DATASET/fathfullness/rerun"

run_faithfulness_norag_paper_run(prior_rag_dirs=PRIOR_RAG)
```

Expected log:

```text
[faithfulness] NoRAG paper env: context=task_grounding ...
[faithfulness] NoRAG work rows=18180; 37 batches @ 500
```

Re-run **Cell 3** after each batch until:

```text
[faithfulness] All 37 batches complete (18180/18180 NoRAG rows scored)
```

Outputs: `/kaggle/working/Faithfulness/norag/*_batch_*.csv`

---

## 3. Merge RAG + NoRAG final CSV

Copy or symlink RAG batches under one tree, or point `build_deduped_faithfulness_final` at `/kaggle/working/Faithfulness` (recursive glob):

```python
import os
os.environ["GP_FAITH_TARGET_ROWS"] = "all"  # expect 36,360 rows

build_deduped_faithfulness_final(
    "/kaggle/working/Faithfulness",
    output_path="/kaggle/working/Faithfulness/benchmark_results_all_predictions_combined_faithfulness_final.csv",
)
```

Download the final CSV and run locally:

```powershell
cd "d:\Green Paper\GreenPaper_Kaggle_Benchmarks"
python build_paper_artifacts.py
```

Then update Table 4 in `results.tex` with `SLM_NoRAG` / `LLM_NoRAG` faithfulness cells.

---

## 4. Resume from batch N

```python
import os
os.environ["GP_FAITH_START_BATCH"] = "11"  # 1-indexed
run_faithfulness_norag_paper_run(prior_rag_dirs=PRIOR_RAG)
```

---

## 5. CLI (local)

```powershell
python run_faithfulness_eval.py --predictions result/benchmark_results_all_predictions_combined.csv --batch_size 500 --norag_only
```
