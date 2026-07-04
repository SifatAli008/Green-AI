# Kaggle: gold-aligned PubMedQA RAG index (audit fixes C1–C4)

## Critical fixes in `eval_benchmarks.py` (bundle `20260521a+`)

| Issue | Fix |
|-------|-----|
| **C1** Gold index = eval items | Gold `chunks.jsonl` built from **corpus holdout only** (80% default); eval PMIDs excluded |
| **C2** Oracle retrieval query | Retrieval uses **question only** (not abstract). Env override: `GP_PUBMEDQA_RETRIEVAL_QUERY_INCLUDES_ABSTRACT=1` |
| **C3** PubMedQA train eval | Eval on fixed **holdout** from `pqa_labeled` (not official test); see `pubmedqa_split` in JSON |
| **C4** Dual corpus | `rag_experiment` in results JSON — report MCQ (external) and PubMedQA (gold) **separately** |
| **C5** Invalid generative metrics | PubMedQA: **label accuracy only**; BLEU/ROUGE/BERTScore skipped unless `GP_PUBMEDQA_ENABLE_GENERATIVE_METRICS=1` |
| **C6** MMLU contamination | Default **`--mmlu_split dev`** (not `test`) |

## Step 1 — rebuild gold index (required after upgrade)

Old gold indexes that include eval PMIDs will fail leak check — rebuild:

```bash
python eval_benchmarks.py --build_gold_index /kaggle/working/rag_index_gold
```

Writes `chunks.jsonl`, `index.faiss`, and `pubmedqa_gold_manifest.json`.

## Step 2 — eval

```bash
python /kaggle/working/eval_benchmarks.py --benchmark all --max_items 500 --seed 42 \
  --auto_gold_pubmedqa \
  --mmlu_split dev \
  --out_json /kaggle/working/benchmark_results_dual.json
```

Check JSON fields:

- `pubmedqa_split` — holdout descriptor
- `rag_experiment` — dual corpus reporting
- `retrieval_validation.gold_index_eval_pubid_leak_count` — must be **0**
- `benchmarks.pubmedqa.results.*.pubmedqa_metrics_mode` — `label_accuracy_only`

## Step 3 — verify recall@k

```python
import json
j = json.load(open("/kaggle/working/benchmark_results_dual.json"))
print(j["retrieval_validation"])
print(j["rag_experiment"])
rows = json.load(open("/kaggle/working/benchmark_results_dual_predictions.json"))["rows"]
pub = [r for r in rows if r["benchmark"]=="pubmedqa" and r.get("rag_flag")]
print("query mode:", pub[0].get("pubmedqa_retrieval_query_mode"))
print("recall@1 mean:", sum(float(r["recall_at_1"]) for r in pub)/len(pub))
```

Expect `question_only` and non-zero recall only when gold index is rebuilt without eval PMIDs.

## Paper wording

- **PubMedQA accuracy:** holdout from `pqa_labeled`, not official PubMedQA test.
- **Recall@K:** question-only query, gold corpus excludes eval PMIDs; pipeline QA metric, not clinical retrieval.
- **MMLU-Med:** report split used (`dev` default).
- **RAG:** separate tables for MCQ external RAG vs PubMedQA gold RAG.
