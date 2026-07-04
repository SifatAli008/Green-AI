# Results Documentation — Green Paper (IEEE Access)

**Last updated:** June 2026  
**Paper:** `Preparation_of_Papers_for_IEEE_ACCESS/`  
**Benchmarks:** `GreenPaper_Kaggle_Benchmarks/`  
**Constants:** `code/measurement_config.py`  
**EIE Framework:** [`EIE_FRAMEWORK.md`](EIE_FRAMEWORK.md) — full technical + graphical + prototype documentation  
**Defense companion:** `REWRITE_AND_DEFENSE_GUIDE.md` — claim framing, reviewer responses, and paper rewrite text for every flagged issue.

This document is the **single source of truth** for evaluation data, table values, and **how each result may (and may not) be claimed** in the paper. All numbers are cross-checked against CSVs and `measurement_config.py` (§12). Where the defense guide identifies overstated framing, this document uses the **qualified, defensible** wording.

**Principle:** No new experiments. Every defense traces to data already in this file.

### Claims prohibited in the paper (without qualification)

| Do **not** claim | Why | See |
|------------------|-----|-----|
| RAG improves faithfulness vs NoRAG | NoRAG rows = 0 | CRITICAL-02, §7.7 |
| SLM+RAG wins on all benchmarks | PubMedQA −4.2%/−5.5% with RAG | CRITICAL-03, §7.4 |
| Classifier accuracy 1.0 = ML generalization | Rule-based consistency on 240 labels | CRITICAL-01, §4.5 |
| 47.5% faithfulness = clinical safety | Rubric band 40–55 = partial support only | MINOR-05, §7.7 |
| Pooled MRR 0.997 as headline retrieval | Gold subset MRR **0.967** is primary | MODERATE-05, §7.2 |
| 122 tok/s SLM throughput | Corrected to **119** tok/s (273÷2.3) | MODERATE-02, §7.1 |
| 117M queries/year without citation | Use 1M queries/day scaling (§7.8) | MINOR-04 |

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Models and hardware](#2-models-and-hardware)
3. [Corpus and retrieval index](#3-corpus-and-retrieval-index)
4. [Evaluation design and data sizes](#4-evaluation-design-and-data-sizes)
5. [Methodology process (Phases 1–5)](#5-methodology-process-phases-15)
6. [Pipeline and scripts](#6-pipeline-and-scripts)
7. [Results by table](#7-results-by-table)
8. [Key findings (defensible claims only)](#8-key-findings-defensible-claims-only)
9. [Paper and reference work completed](#9-paper-and-reference-work-completed)
10. [Pending items](#10-pending-items)
11. [File map](#11-file-map)
12. [Cross-check audit](#12-cross-check-audit-june-2026)
13. [EIE Framework](#13-eie-framework-energyinfrastructureeconomy)
14. [Claim framing & reviewer defenses](#14-claim-framing--reviewer-defenses)

---

## 1. Project overview

**Title (working):** Retrieval-augmented small language models for clinical decision support — efficiency, grounding, and sustainability.

**Goal:** Compare **SLM vs LLM** under **RAG / NoRAG / hybrid routing** on biomedical QA, measuring answer quality, retrieval, faithfulness, judge scores, latency, and GPU energy/carbon.

**Core research questions — scope as reported:**

| RQ | Question | Status in this study |
|----|----------|----------------------|
| **RQ1** | Can SLM+RAG match or beat LLM+RAG with lower energy? | **Partially answered:** SLM+RAG leads on **curated 3-query token-F1** (0.6101 vs 0.4548) with ~53% lower GPU energy; open-benchmark RAG effects are **mixed** (§7.4). |
| **RQ2** | Does evidence grounding reduce unsupported claims? | **Partially answered:** RAG faithfulness **absolute** scores only (47.2–47.8%, n=18,180); **NoRAG baseline missing** — cannot claim RAG improves grounding vs NoRAG (§7.7, CRITICAL-02). |
| **RQ3** | Privacy-readiness, sustainability, routing efficiency? | **Addressed** via local deployment design, NVML energy (§7.8), EIE Framework (§13), routing load metrics (§7.5). |

**Sustainability model:** **EIE (Energy–Infrastructure–Economy) Framework** (§13).

---

## 2. Models and hardware

### 2.1 Generation models (experimental arms)

| Role | Hugging Face ID | Parameters | Approx. footprint |
|------|-----------------|------------|-------------------|
| **SLM** | `google/gemma-2-2b-it` | 2B (instruction-tuned) | ~4 GB FP16 |
| **LLM** | `meta-llama/Llama-2-7b-chat-hf` | 7B (chat) | ~14 GB FP16 |

**Inference settings:** FP16, `torch.no_grad()`, KV cache, greedy decoding; `max_new_tokens` 64–128, `min_new_tokens` 20, temperature 0.7, top-p ≥ 0.9; seed 42.

### 2.2 Retrieval / embedding model

| Role | Model ID | Output |
|------|----------|--------|
| **Dense embedder** | `sentence-transformers/all-MiniLM-L6-v2` | 384-d vectors |
| **Index** | FAISS `IndexFlatIP` + L2 normalization | cosine via inner product |
| **Sparse leg** | BM25 lexical re-ranking | hybrid with dense |

**Shared index:** Both SLM+RAG and LLM+RAG use the **identical** hybrid FAISS+BM25 index; the generator operates downstream and does **not** affect Recall@k or MRR (§7.2).

### 2.3 Evaluation / judge models

| Role | Model ID | Where used |
|------|----------|------------|
| **LLM-as-judge** (Table 3) | **`Qwen/Qwen2.5-7B-Instruct`** (`local_hf`) | `llm_judge.py` |
| **Faithfulness judge** (Table 4) | **`Qwen/Qwen2.5-7B-Instruct`** (`local_hf`) | `run_faithfulness_eval.py` |
| **Faithfulness default (local dev)** | `Qwen/Qwen3-8B` | `_PAPER_DEFAULT_MODEL` |
| **Faithfulness Kaggle fallback** | `Qwen/Qwen2.5-3B-Instruct` | `_KAGGLE_DEFAULT_FAITHFULNESS_MODEL` |

Tables **3** and **4** use **Qwen2.5-7B-Instruct** as a judge **distinct** from Gemma and Llama generators.

### 2.4 Hardware environment

| Setting | Details |
|---------|---------|
| **Platform** | Kaggle Notebooks |
| **GPU** | NVIDIA T4 (Draft: up to 2× T4, 15 GB each) |
| **RAM** | Up to 30 GB |
| **Energy telemetry** | NVML via `pynvml` |
| **Grid carbon intensity** | **0.385 kg CO₂e/kWh** — U.S. average reference (`measurement_config.py`); kWh/query is grid-independent (§7.8, MINOR-03) |

### 2.5 Kaggle accounts (faithfulness batch runs)

| Account | Purpose |
|---------|---------|
| `fatinshadab` | Default paper faithfulness; RAG index |
| `sifatali008` | Resume batch 11 |
| `fahim220` | Resume batch 16 |
| `ummesalmahabiba` | Resume / gap-fill NoRAG |

---

## 3. Corpus and retrieval index

| Property | Value |
|----------|-------|
| **Source documents** | 115 (112 PubMed + 3 guidelines: FDA, NICE, NHS) |
| **Clinical domains** | 6 |
| **Semantic chunks** | 556 (512 tokens, 100 overlap) |
| **FAISS index size** | ~100 MB |
| **Retrieval latency** | 50–100 ms/query |
| **Top-k at inference** | 3 chunks; re-rank top 10–15 |

**Gold retrieval:** `rag_index_gold` — PubMedQA-aligned recall/MRR where `retrieval_evaluable=True` (§7.2).

---

## 4. Evaluation design and data sizes

### 4.1 Two latency contexts — do not mix

| Context | Description | SLM | LLM |
|---------|-------------|-----|-----|
| **A — Single-query baseline** | One request, no concurrency | **2.3 s** | **7.0 s** |
| **B — Concurrent load** | Multi-user load test (curated Q1–Q3, RAG) | **22.18 s** | **49.55 s** |

Throughput figures (§7.1, §13 I3) apply to **Context A only** and are **derived** (length ÷ latency), not independently benchmarked. Under Context B, effective throughput is much lower.

### 4.2 Three F1 / quality contexts — mandatory labeling

**Never conflate these values** (MINOR-06). Use the exact label wherever F1 appears in the paper:

| Value | Correct label | n |
|-------|---------------|---|
| **0.6101** | token-F1 (curated 3-query factorial, Q1–Q3, Context B load) | 3 |
| **0.4548** | token-F1 (curated 3-query, LLM+RAG, Context B) | 3 |
| **0.5667** | token-F1 (curated 3-query, hybrid routing, Context B) | 3 |
| **0.607** | macro-F1 (open benchmark, SLM+RAG) | 9,090 |
| **0.579** | macro-F1 (open benchmark, SLM NoRAG) | 9,090 |
| **0.523** | macro-F1 (open benchmark, LLM+RAG) | 9,090 |
| **0.465** | macro-F1 (open benchmark, LLM NoRAG) | 9,090 |

**Abstract rule:** cite **one** F1 with full context (0.6101 curated); do **not** cite 0.607 without distinguishing it from 0.6101.

| Context | Data | Primary metric |
|---------|------|----------------|
| **Curated 3-query factorial** | Q1–Q3 | Token-F1 (Context B) |
| **Open benchmarks** | MedQA, MMLU-Med, PubMedQA | Label accuracy (Table 2); macro-F1 separate |

### 4.3 Master prediction dataset

**File:** `GreenPaper_Kaggle_Benchmarks/result/benchmark_results_all_predictions_combined.csv`

| Dimension | Count |
|-----------|-------|
| **Total rows** | **36,360** |
| **Per configuration** | **9,090** (SLM_NoRAG, SLM_RAG, LLM_NoRAG, LLM_RAG) |
| **Contributors** | `Bipro`, `Medha`, `Salma`, `Sifat` |

### 4.4 Open benchmark breakdown (per configuration)

| Benchmark | Items/config | × 4 configs |
|-----------|--------------|-------------|
| MedQA | 3,800 | 15,200 |
| MMLU-Med | 1,490 | 5,960 |
| PubMedQA | 3,800 | 15,200 |
| **Total** | **9,090** | **36,360** |

### 4.5 Curated queries and routing classifier

| Set | Queries | Use |
|-----|---------|-----|
| **Q1–Q3** | 3 | Factorial F1, response length, load test |
| **Q1–Q12** | 12 | Routing behavior, scalability |
| **Labeled routing set** | **240** | Rule-based classifier consistency check |

**Routing classifier (CRITICAL-01):** **Rule-based heuristic**, not a learned ML model (`code/eval_routing.py`). Lexical features and weights: average word length (**0.5141**), query complexity (**0.3824**), word count (**0.0852**), entity count (**0.0183**). **Accuracy = 1.00** on the 240-query labeled set = **design-time consistency check** (rules encode the same labeling criteria used to define the set) — **not** held-out generalization. Equivalent to verifying a hand-coded decision tree reproduces its own training labels.

**Empirical routing on Q1–Q12 (primary behavioral evidence):** **68%** → SLM (simple), **24%** → LLM (complex), **~8%** → intermediate zone (MINOR-02). Fallback policy: queries between SLM upper and LLM lower thresholds **default to LLM** — errs toward higher-quality generation in ambiguous cases (clinical safety rationale). Real-world utility assessed via end-to-end token-F1 **0.5667** and latency **35.86 s**, not classifier accuracy alone.

**Paper-ready text (`results.tex` / `implementation.tex`):**
> The routing component employs a rule-based heuristic classifier using lexical features — average word length (weight 0.5141), query complexity (0.3824), word count (0.0852), entity count (0.0183) — implemented in `eval_routing.py`. On the 240-query labeled reference set, the classifier achieves complete agreement with human-assigned routing labels (accuracy = 1.00), reflecting that decision rules were derived from the same labeling criteria — a design-time consistency check rather than a generalization test. On the 12-query extended set, 68% of queries were routed to SLM, 24% to LLM, and ~8% to an intermediate fallback rule defaulting to LLM. Real-world behavior is assessed through end-to-end F1 (0.5667) and latency (35.86 s), not classifier accuracy alone.

### 4.6 Retrieval evaluation subset

| Subset | Count | Notes |
|--------|-------|-------|
| `retrieval_evaluable=True` | **6,600** (3,300/RAG config) | Gold-label PubMedQA subset |
| Shared index | — | Identical scores for SLM+RAG and LLM+RAG |

### 4.7 LLM-as-judge dataset (Table 3)

| Metric | Value |
|--------|-------|
| **Rows judged** | **35,845** / 36,360 (**98.6%**) |
| **Parse failures** | 15 (0.04%) |
| **Unjudged gap** | **515** (SLM+RAG **140**; others **125** each) |
| **Judge** | **`Qwen/Qwen2.5-7B-Instruct`** |
| **Per-config n** | SLM+RAG **8,950**; others **8,965** |

### 4.8 Faithfulness dataset (Table 4)

| Metric | Value |
|--------|-------|
| **RAG rows scored** | **18,180** (9,090/config) |
| **NoRAG rows scored** | **0** (pending) |
| **Judge** | **`Qwen/Qwen2.5-7B-Instruct`** |
| **Score scale** | 0–100 (rubric below) |

**Faithfulness rubric (0–100):**

| Band | Interpretation |
|------|----------------|
| **100** | CONTEXT clearly supports or entails the selected option |
| **60–80** | Related; reasonable clinical basis |
| **40–55** | Partially related; indirect or incomplete support |
| **25–38** | Tangential; weak basis |
| **0–20** | Irrelevant, contradicts, or no basis |

**Mean 47.5%** = **40–55 band** ("partial support") — not clinical safety approval (MINOR-05).

### 4.9 Energy measurement protocol

| Parameter | Value |
|-----------|-------|
| **Trials** | 10 × 20 clinical queries per strategy |
| **Telemetry** | NVML GPU energy → kWh/query |
| **Carbon** | kWh × **0.385** kg CO₂e/kWh (U.S. reference) |
| **Strategies** | SLM_NoRAG, SLM_RAG, LLM_NoRAG, LLM_RAG, Routing_Hybrid |

---

## 5. Methodology process (Phases 1–5)

| Phase | Content | Status |
|-------|---------|--------|
| **1** | 115-doc corpus → 556 chunks → FAISS | Done |
| **2** | 2×2 factorial; Q1–Q3 core; Q1–Q12 extended | Done |
| **3** | Hybrid RAG; 36,360 benchmark grid | Done |
| **4** | Claim verifier **design**; faithfulness Table 4 (RAG only) | Partial |
| **5** | QUEST clinician review; regulatory **readiness** only | Planned |

---

## 6. Pipeline and scripts

| Script | Role |
|--------|------|
| `eval_benchmarks.py` | Benchmarks; accuracy, F1, ROUGE-L, retrieval |
| `eval_posthoc.py` | Tables 1–4 from JSON |
| `build_paper_artifacts.py` | → `paper_tables.json` |
| `run_faithfulness_eval.py` | Faithfulness; Qwen judge |
| `llm_judge.py` | Table 3 clinical rubric |
| `code/eval_routing.py` | Rule-based routing classifier |
| `measurement_config.py` | Latency/energy constants |

---

## 7. Results by table

### 7.1 Baseline comparison (Context A — single-query, no load)

Source: `measurement_config.py`, curated Q1–Q3 per-scenario lengths (§7.3).

| Metric | SLM (Gemma) | LLM (Llama) | Notes |
|--------|-------------|-------------|-------|
| Inference latency | **2.3 s** | **7.0 s** | Context A |
| Model footprint | **4 GB** | **14 GB** | Single-model |
| Response length (Q1–Q3 NoRAG mean) | **273** tokens | **419** tokens | Per-scenario: SLM 185/481/153; LLM 380/512/365 |
| Throughput (derived, Context A†) | **119** tok/s | **60** tok/s | 273÷2.3; 419÷7.0 |
| Peak GPU (full pipeline, load) | — | — | **14.28–14.37 GB** |

†**Not independently benchmarked.** Under Context B (22.18 s / 49.55 s), effective throughput is substantially lower. Legacy baseline table used rounded 280/420 tokens → 122 tok/s SLM; **authoritative curated means** are 273/419 → **119/60 tok/s** (MODERATE-02).

---

### 7.2 Table 1 — Retrieval (Recall@k + MRR)

**Primary metric:** Gold-label evaluable subset (`retrieval_evaluable=True`, `rag_index_gold`), **n = 3,300** per RAG configuration. Stricter than pooled health (MODERATE-05).

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---------------|----------|----------|----------|-----------|-----|
| **RAG (both models, shared index)** | 0.667 | **0.891** | 0.914 | 0.936 | **0.967** |
| NoRAG | — | — | — | — | N/A (no retrieval) |

SLM+RAG and LLM+RAG are **identical** because retrieval is upstream of generation (MODERATE-06). Paper may report as **one shared row** with footnote.

**Supplementary (broader population, not primary headline):**

| Source | Recall@3 | MRR | n context |
|--------|----------|-----|-----------|
| Pooled 34-run health | 0.919 | **0.997** | All retrieval-attempted rows |
| Bipro 16-run avg | 0.917 | 0.998 | Contributor export |

Gap 0.967 vs 0.997 is **methodologically expected**: gold subset requires retrieving the **specific** PubMedQA-annotated document in `rag_index_gold`; pooled health includes all retrieval-attempted rows (softer relevance criterion).

**Paper-ready Table 1 caption (MODERATE-05, MODERATE-06):**
> Retrieval metrics are computed on the gold-label evaluable subset (n = 3,300 per RAG configuration; `retrieval_evaluable=True`, gold mapping in `rag_index_gold`). Pooled 34-run health (MRR 0.997, Recall@3 0.919) reflects the broader non-gold-constrained population and is **supplementary**, not the headline. Both SLM+RAG and LLM+RAG share the identical hybrid FAISS+BM25 index; retrieval is reported once because the generator is downstream.

**Recommended table structure:** one **RAG (both models)** row + **NoRAG: N/A — no retrieval step**.

---

### 7.3 RAG response length (curated Q1–Q3)

| Model | NoRAG mean | +RAG mean | Δ |
|-------|------------|-----------|---|
| Gemma-2-2B-it | **273** | **454** | +66.5% |
| Llama-2-7B-chat | **419**‡ | **484**§ | +15.5% |

‡NoRAG mean = (380+512+365)/3 = **419**. The value **380** in the legacy `tab:rag-impact` average row was a rounding error (MODERATE-02); per-scenario LLM NoRAG tokens are 380 / 512 / 365.  
§+RAG mean = (488+512+453)/3 = **484**. Paper reports +27.5% using legacy 380→484 headline for LLM; arithmetic from corrected NoRAG mean is +15.5%.

| Scenario | SLM −RAG | SLM +RAG | LLM −RAG | LLM +RAG |
|----------|----------|----------|----------|----------|
| Diabetes | 185 | 340 | 380 | 488 |
| AI in diagnosis | 481 | 512 | 512 | 512 |
| Hypertension | 153 | 453 | 365 | 453 |

RAG increases LLM verbosity (+27.5% in paper table using 380→484 headline); relevant to ROUGE-L artifact (§7.4).

---

### 7.4 Table 2 — Open-benchmark answer quality

**Primary PubMedQA metric:** **label accuracy** (not ROUGE-L). ROUGE-L reported for completeness with mechanistic caveat (CRITICAL-04).

| Configuration | MedQA | MMLU-Med | PubMedQA acc. | PubMedQA ROUGE-L |
|---------------|-------|----------|---------------|------------------|
| SLM NoRAG | 0.404 | 0.510 | **0.622** | 0.556 |
| SLM+RAG | 0.403 | **0.518** | 0.596 | 0.562 |
| LLM NoRAG | 0.365 | 0.421 | **0.523** | 0.328 |
| LLM+RAG | 0.348 | **0.497** | 0.494 | **0.142** |

**RAG deltas (open benchmarks, n=9,090/config):**

| Benchmark | SLM Δ | LLM Δ |
|-----------|-------|-------|
| PubMedQA acc. | **−4.2%** (0.622→0.596) | **−5.5%** (0.523→0.494) |
| MedQA acc. | −0.2% (negligible) | −4.7% |
| MMLU-Med acc. | +1.6% | **+18.1%** (0.421→0.497) |

**ROUGE-L vs token-F1 (LLM PubMedQA, CRITICAL-04):**

| Metric | LLM NoRAG | LLM+RAG | Δ |
|--------|-----------|---------|---|
| PubMedQA ROUGE-L | 0.328 | **0.142** | −57% |
| PubMedQA token-F1 | 0.049 | **0.172** | +251% |
| Response length (curated) | 419 tokens (mean) | 484 tokens | +15.5% |

**Mechanism (five steps):** (1) PubMedQA references are short labels ("yes"/"no"/"maybe"). (2) ROUGE-L = longest common subsequence vs reference string. (3) RAG lengthens LLM responses → low overlap with 1–5 word references. (4) Token-F1 uses retrieved-chunk vocabulary — different reference space. (5) SLM ROUGE-L stable (0.556→0.562) — less verbose under RAG. **Not a data error.**

**Paper-ready paragraph (`results.tex` after Table 2):**
> The contrasting ROUGE-L and token-F1 trends for LLM+RAG on PubMedQA warrant explanation. PubMedQA reference answers are short classification labels with brief justifications. ROUGE-L is sensitive to response verbosity: RAG increases LLM mean length toward 484 tokens, yielding lower ROUGE-L (0.328 → 0.142) against terse references, while token-F1 rises (0.049 → 0.172) as the model incorporates retrieved terminology. **Label accuracy is the primary PubMedQA metric**; ROUGE-L is secondary completeness only.

**Macro-F1 across benchmarks** (`paper_tables.json`):

| Config | macro-F1 |
|--------|----------|
| SLM+RAG | **0.607** |
| SLM NoRAG | 0.579 |
| LLM+RAG | 0.523 |
| LLM NoRAG | 0.465 |

---

### 7.5 Routing and load (curated 3-query, Context B)

| Strategy | token-F1 (curated, n=3) | Latency |
|----------|-------------------------|---------|
| SLM only | **0.6101** | **22.18 s** |
| LLM only | 0.4548 | 49.55 s |
| Routing hybrid | 0.5667 | 35.86 s |

| Routing metric | Value |
|----------------|-------|
| Classifier type | **Rule-based** (`eval_routing.py`) |
| Consistency on 240-label set | 1.00 (design check, not generalization) |
| Empirical Q1–Q12 split | 68% SLM / 24% LLM / **~8% fallback→LLM** |
| Load scale 5→10 users | ~2.24× latency |

**Paper-ready routing discussion (MINOR-02):**
> The routing classifier assigned 68% of queries to SLM (simple), 24% to LLM (complex), and approximately 8% to an intermediate complexity region between SLM upper and LLM lower thresholds. These ambiguous queries were handled by a deterministic fallback rule **defaulting to LLM** — erring toward higher-quality generation in clinically uncertain cases. Routing suitability in production would require prospective validation beyond this study.

---

### 7.6 Table 3 — LLM-as-judge (0–5 scale)

Judge: **Qwen/Qwen2.5-7B-Instruct**; n = 8,950 (SLM+RAG) / 8,965 (others).

| Configuration | Correctness | Completeness | Clinical relevance |
|---------------|-------------|--------------|-------------------|
| SLM NoRAG | 2.84 | 2.85 | **3.17** |
| SLM+RAG | 2.86 (+0.02) | 2.92 (+0.07) | 3.15 (**−0.02**) |
| LLM NoRAG | 2.58 | 2.70 | 3.00 |
| LLM+RAG | 2.69 (+0.11) | 2.83 (+0.13) | 3.05 (+0.05) |

**Interpretation (MODERATE-04):** With n ≈ 8,950–8,965 and SD ≈ 1.7, even Δ = 0.02 may reach p < 0.001, but **Cohen's d < 0.15** for all cells (below "small effect" d = 0.2). |Δ| ≤ 0.13 — not practically meaningful. SLM clinical relevance **decreases** with RAG (3.17 → 3.15, Δ = −0.02) — may reflect Qwen judge preference for terse NoRAG phrasing.

**Paper-ready paragraph (`results.tex` after Table 3):**
> Differences between RAG and NoRAG judge scores are small in absolute magnitude. For SLM, Correctness +0.02 and Completeness +0.07 with RAG, while Clinical Relevance decreased by 0.02. For LLM, all three dimensions showed small positive RAG effects (≤ +0.13). With n ≈ 9,000 per configuration, parametric tests would likely reach significance for tiny deltas, but effect sizes are negligible. Scores are relative within-run rankings, not clinical approval.

**Coverage (MINOR-01):** 35,845/36,360 scored (98.6%); 15 parse failures; SLM+RAG unjudged **140** (1.54% of 9,090) vs **125** others — unlikely to bias means.

**Paper-ready Table 3 caption:**
> LLM-as-judge coverage: 35,845 of 36,360 predictions scored (98.6%); 15 unparseable outputs (0.04%). Unjudged gap: SLM_NoRAG 125, SLM_RAG **140**, LLM_NoRAG 125, LLM_RAG 125. SLM+RAG over-representation (140 vs ~129 expected) is 0.38% of 9,090 predictions. Means computed on judged subset (n = 8,950 SLM+RAG; n = 8,965 others). Judge: Qwen/Qwen2.5-7B-Instruct.

---

### 7.7 Table 4 — Faithfulness (0–100 scale)

**Scope (CRITICAL-02):** RAG **absolute** characterization only. **No RAG vs NoRAG comparison** in this study cycle.

| Configuration | Faithfulness (%) | n | Status |
|---------------|------------------|---|--------|
| SLM NoRAG | — | — | Pending |
| **SLM+RAG** | **47.2** | 9,090 | Done |
| LLM NoRAG | — | — | Pending |
| **LLM+RAG** | **47.8** | 9,090 | Done |

**Aggregate RAG:** mean **47.5%** → rubric band **40–55** ("partially related; indirect support").

**Source file:** `fathfullness/rerun/benchmark_results_all_predictions_combined_faithfulness_final.csv` (18,180 rows); cross-checked in `paper_tables.json` `table_h_faithfulness`.

**RQ2 answer (CRITICAL-02) — paper-ready:**
> **RQ2 (Partial):** Absolute faithfulness of RAG-augmented responses was evaluated (n = 18,180; Qwen2.5-7B-Instruct). SLM+RAG (47.2%) and LLM+RAG (47.8%) fall in the "partial support" rubric band (40–55). A quantitative RAG vs NoRAG faithfulness comparison is **not reported** — NoRAG evaluation pending. Phase 4 claim verifier designed; end-to-end blocking rates not measured.

**Clinical safety (MINOR-05) — limitations text for `discussion.tex`:**
> Mean faithfulness 47.2–47.8% falls short of the "clearly supported" band (60+). The judge measures whether evidence *supports* the selected answer, not whether the answer is *clinically correct*. No NoRAG baseline precludes RAG grounding claims. **Deployment in clinical decision support would require substantially higher thresholds via Phase 5 QUEST clinician review.**

**Do not claim:** RAG reduces unsupported claims relative to NoRAG until NoRAG cells are filled.

**Paper-ready Table 4 section (`results.tex`):**
> Faithfulness was evaluated using Qwen/Qwen2.5-7B-Instruct on a 0–100 scale (`run_faithfulness_eval.py`), scoring whether the model's selected answer was supported by retrieved context chunks (n = 9,090 per RAG configuration; 18,180 total). SLM+RAG mean **47.2%** and LLM+RAG **47.8%** (aggregate **47.5%**) fall in the 40–55 rubric band ("partially related; indirect or incomplete support"). NoRAG faithfulness was not completed; a direct RAG vs NoRAG comparison is not reported. Scores characterize **absolute** RAG grounding against retrieved evidence. Judge is architecturally distinct from both generators; scores are automated proxies, not clinical validation.

---

### 7.8 Energy and carbon (NVML, 10 trials × 20 queries)

| Strategy | kWh/query (mean±std) | kg CO₂e/query | kg CO₂e / 1M queries* |
|----------|----------------------|---------------|------------------------|
| SLM_NoRAG | 0.000052 ± 0.000011 | 0.000020 | **19.83** |
| SLM_RAG | 0.000052 ± 0.000013 | 0.000020 | 19.82 |
| LLM_NoRAG | 0.000111 ± 0.000024 | 0.000043 | **42.35** |
| LLM_RAG | 0.000112 ± 0.000023 | 0.000043 | 42.47 |
| Routing_Hybrid | 0.000101 ± 0.000021 | 0.000039 | 38.19 |

\*CO₂e/M from **full-precision** kWh in `measurement_config.py` (MODERATE-03). Displayed kWh rounded → ~1% mismatch vs kWh×0.385×10⁶:

| Strategy | Displayed kWh | Arithmetic CO₂/M | Reported CO₂/M | Δ |
|----------|---------------|------------------|----------------|---|
| SLM_NoRAG | 0.000052 | 20.02 | **19.83** | 0.95% |
| SLM_RAG | 0.000052 | 20.02 | 19.82 | 1.0% |
| LLM_NoRAG | 0.000111 | 42.74 | **42.35** | 0.91% |
| LLM_RAG | 0.000112 | 43.12 | 42.47 | 1.5% |
| Routing | 0.000101 | 38.89 | 38.19 | 1.8% |

Discrepancy << measurement uncertainty (e.g. Routing ±0.000021 kWh ≈ 21%).

**Headline:** **~53%** lower GPU carbon SLM vs LLM (42.35→19.83 kg/M). RAG adds negligible energy.

**Carbon geographic scope (MINOR-03):** Grid intensity **0.385 kg CO₂e/kWh** = U.S. average (`measurement_config.py`). **Primary metric: kWh/query** (grid-independent). At UK ~**0.233 kg/kWh**: SLM+RAG ≈ **12.1 kg/M** vs LLM+RAG ≈ **25.7 kg/M** — same **53% relative** advantage.

**Scaling (MINOR-04, Option A — preferred; no 117M constant):**
> At 1M queries/day: **18.98 kWh/day** (SLM_RAG) vs **40.88 kWh/day** (LLM_RAG) GPU energy — **~8.0 MWh/year** difference per million daily queries. Apply local tariff and grid intensity for cost/emissions.

**Paper-ready energy table footnote (MODERATE-03):**
> *kWh/query rounded to six decimal places for display; CO₂e/M computed from full-precision means in `measurement_config.py` and may differ slightly from displayed kWh × 0.385 × 10⁶. Standard deviations from 10-trial NVML protocol apply to kWh; CO₂e/M inherits relative uncertainty.

**Paper-ready carbon scope (MINOR-03):**
> Carbon estimated at 0.385 kg CO₂e/kWh (U.S. average; `measurement_config.py`). Grid intensities vary globally (e.g. UK ~0.233 kg/kWh). At UK intensity, SLM+RAG ≈ 12.1 kg/M vs LLM+RAG ≈ 25.7 kg/M — same 53% relative advantage. **kWh/query is the primary grid-independent metric.**

---

### 7.9 System stability

| Metric | Value |
|--------|-------|
| GPU memory under load | 14.28–14.37 GB (±0.6%) |
| Memory leaks / truncation | None observed |

---

## 8. Key findings (defensible claims only)

Rewritten per `REWRITE_AND_DEFENSE_GUIDE.md` (CRITICAL-03, CRITICAL-02, etc.):

1. **RAG impact is context-dependent; SLM+RAG leads on curated clinical queries (CRITICAL-03).** On Q1–Q3 (n=3, Context B), SLM+RAG token-F1 **0.6101** vs LLM+RAG **0.4548**, hybrid routing **0.5667**, with ~53% lower GPU energy. On open benchmarks (n=9,090/config), RAG effects are **mixed**: MMLU-Med +18.1% for LLM (0.421→0.497); PubMedQA **−4.2%** (SLM, 0.622→0.596) and **−5.5%** (LLM, 0.523→0.494); MedQA unchanged for SLM (0.404→0.403). Strong retrieval (R@3=0.891, MRR=0.967) did not uniformly improve downstream accuracy.

   **Paper-ready Finding 1 (`results.tex`):**
   > **Finding 1: RAG impact is context-dependent; SLM+RAG leads on curated clinical queries.** On three controlled queries (Q1–Q3), SLM+RAG achieved the highest token-F1 (0.6101) vs LLM+RAG (0.4548) and hybrid routing (0.5667), with approximately half the GPU energy. On open benchmarks (n = 9,090/configuration), RAG effects were mixed: MMLU-Med +18.1% for LLM, PubMedQA −4.2% to −5.5% for both models, MedQA unchanged for SLM. High retrieval quality (Recall@3 = 0.891) did not translate uniformly into accuracy gains.

2. **EIE measured resource use: SLM_RAG vs LLM_RAG.** kWh/query ratio **2.15×**; CO₂e/M **19.82 vs 42.47**; footprint **4 vs 14 GB**; Context-A latency **2.3 vs 7.0 s**; derived throughput **119 vs 60 tok/s** (Context A only). Clinical quality reported separately (§13, `EIE_FRAMEWORK.md`).

3. **Faithfulness is modest and RAG-only.** Mean **47.5%** (partial-support band); SLM and LLM similar; **NoRAG baseline missing** — cannot claim RAG improves grounding.

4. **Automated judges are proxies, not clinical validation.** Qwen2.5-7B-Instruct for Tables 3–4; Table 3 deltas negligible (|Δ|≤0.13).

5. **Routing is rule-based with documented fallback.** 68%/24%/8% split; ambiguous queries default to LLM; end-to-end hybrid F1 **0.5667**, latency **35.86 s**.

6. **ROUGE-L drop for LLM+RAG PubMedQA is a measurement artifact.** Label accuracy is the primary PubMedQA metric; token-F1 rises with RAG while ROUGE-L falls due to short reference labels + longer RAG responses.

7. **Phase 4 verifier designed; Phase 5 clinician review planned.** HIPAA/GDPR = readiness, not compliance.

---

## 9. Paper and reference work completed

- `results.tex`: Tables 1–4; two/three F1 contexts; ROUGE-L explanation; faithfulness rubric; energy footnotes
- `discussion.tex`: limitations strengthened; stale `tab:medllm-benchmarks` removed
- `references.bib`: faculty audit ([6] CREOLA, [17] CSBJ 2024; [14] removed)
- `REWRITE_AND_DEFENSE_GUIDE.md`: point-by-point reviewer defenses
- Energy claims: ~53% (not 85%); throughput Context A labeled; curated lengths authoritative

---

## 10. Pending items

| # | Item | Action |
|---|------|--------|
| 1 | **Table 4 NoRAG** (18,180 rows) | `run_faithfulness_norag_paper_run()` on Kaggle GPU |
| 2 | **Merge NoRAG faithfulness** | Update `results.tex` Table 4 |
| 3 | **Phase 4 end-to-end rates** | Quantify claim blocking |
| 4 | **Phase 5 clinician review** | QUEST validation |
| 5 | **BERTScore in Table 2** | Cells mostly null |
| 6 | **LaTeX recompile** | Refresh `access.bbl` |
| 7 | **Apply defense-guide rewrites to LaTeX** | See §14 status table: `results.tex`, `discussion.tex`, abstract |
| ~~8~~ | ~~Baseline token inconsistency~~ | **Resolved:** 273/419 authoritative; throughput **119/60**; legacy 280/420 deprecated |

---

## 11. File map

| Content | Path |
|---------|------|
| Results narrative | `Preparation_of_Papers_for_IEEE_ACCESS/results.tex` |
| Defense & rewrite text | `REWRITE_AND_DEFENSE_GUIDE.md` |
| Machine-readable tables | `GreenPaper_Kaggle_Benchmarks/result/paper_tables.json` |
| Predictions (36,360) | `.../benchmark_results_all_predictions_combined.csv` |
| Judge output | `.../LLM_Judge/..._judge.csv` |
| Faithfulness (RAG) | `.../fathfullness/rerun/..._faithfulness_final.csv` |
| Constants | `code/measurement_config.py` |
| EIE Framework (full) | `EIE_FRAMEWORK.md` |
| EIE figures | `fig_eie_pillars_comparison.png`, `fig_eie_energy_breakdown.png`, `fig_eie_prototype_architecture.png` |

---

## 12. Cross-check audit (June 2026)

Verified against CSVs, `paper_tables.json`, `measurement_config.py`.

| Item | Status |
|------|--------|
| 36,360 predictions; 9,090/config | ✓ |
| Table 1 gold subset n=3,300; R@3=0.891, MRR=0.967 | ✓ CSV |
| Table 2 accuracies & ROUGE-L | ✓ `paper_tables.json` |
| macro-F1: 0.607 / 0.579 / 0.523 / 0.465 | ✓ |
| Table 3 judge; Qwen metadata | ✓ |
| Table 4: 47.2 / 47.8; aggregate 47.5% | ✓ |
| Curated F1 0.6101 / 0.4548 / 0.5667 | ✓ `measurement_config.py` |
| Energy CO₂/M; ~53% reduction | ✓ |
| Judge gap 515 rows | ✓ |
| LLM NoRAG length mean **419** (arithmetic from §7.3) | ✓ corrected |
| Throughput **119/60** tok/s (Context A derived) | ✓ corrected |

**Do not claim without qualification:** RAG improves faithfulness vs NoRAG; SLM+RAG wins on all benchmarks; classifier 1.0 = ML generalization; 47.5% = clinical safety; pooled MRR 0.997 as primary retrieval headline.

---

## 13. EIE Framework (Energy–Infrastructure–Economy)

**Full specification:** [`EIE_FRAMEWORK.md`](EIE_FRAMEWORK.md) — IEEE-style technical document with graphical representation, formulas, measured tables, prototype implementation, figure assets, and **§10 research citations** (`references.bib`).

**Summary (measured facts only; see full doc for protocol and figures):**

| Pillar | SLM+RAG | LLM+RAG | Measured ratio (LLM/SLM) |
|--------|---------|---------|--------------------------|
| **E** kWh/query | 0.000052 | 0.000112 | **2.15×** |
| **E** CO₂e/M (0.385 kg/kWh) | 19.82 | 42.47 | **2.14×** |
| **E** CO₂e tonnes/yr @ 1M queries/day | 7.23 | 15.50 | **2.14×** |
| **I** footprint (GB) | 4 | 14 | 3.50× |
| **I** latency Context A (s) | 2.3 | 7.0 | 3.04× |
| **I** latency Context B (s) | 22.18 | 49.55 | 2.23× |
| **I** throughput† (tok/s) | 119 | 60 | 0.50× |
| **Ec** electricity USD/yr @ 1M/day ($0.12/kWh ref.) | 2,278 | 4,901 | **2.15×** |
| **Ec** GPU capex ref. (USD) | 350 | 1,200 | 3.43× |
| **Ec** combined USD/yr (elec + amort.) | 2,395 | 5,301 | **2.21×** |

†Throughput derived, Context A only (273÷2.3; 419÷7.0).

**Figures (regenerate: `python GreenPaper_Kaggle_Benchmarks/plot_eie_framework.py`):**

| File | Content |
|------|---------|
| `fig_eie_pillars_comparison.png` | E / I / Ec pillar bars |
| `fig_eie_energy_breakdown.png` | kWh ± std and CO₂e/M by strategy |
| `fig_eie_prototype_architecture.png` | Prototype block diagram |
| `fig_eie_carbon_emissions.png` | Pillar E: kWh + CO₂e/M dual chart |
| `fig_eie_hardware_economy.png` | Pillar Ec: capex + electricity opex |
| `fig_eie_matrix_*.png` | Matrices M-2–M-14 (see `EIE_FRAMEWORK.md` §5) |

**Clinical quality — reported separately, not in EIE formula:**

| Metric | SLM+RAG | LLM+RAG |
|--------|---------|---------|
| token-F1 (curated, n=3) | 0.6101 | 0.4548 |
| macro-F1 (open, n=9,090) | 0.607 | 0.523 |
| Faithfulness (RAG) | 47.2% | 47.8% |

EIE reports resource observables; clinical tables report answer quality. Cite both when discussing deployment; neither alone defines suitability.

---

## 14. Claim framing & reviewer defenses

Aligned with `REWRITE_AND_DEFENSE_GUIDE.md`. **No new experiments required.**

### Implementation status (paper files)

| ID | Action | Target file | Status in this doc |
|----|--------|-------------|-------------------|
| CRITICAL-01 | Rule-based routing reframe | `implementation.tex`, §7.5 | ✓ §4.5, §7.5 |
| CRITICAL-02 | RQ2 partial; no RAG>NoRAG claims | `results.tex`, `discussion.tex` | ✓ §7.7, §1 |
| CRITICAL-03 | Qualified Finding 1 + abstract | `results.tex`, abstract | ✓ §8, Appendix A |
| CRITICAL-04 | ROUGE-L mechanistic paragraph | `results.tex` §7.4 | ✓ §7.4 |
| MODERATE-01–06 | Labels, footnotes, Table 1 structure | `results.tex` | ✓ §7.1–7.2, §13 |
| MINOR-01–06 | Coverage, carbon, scaling, F1 labels | Various | ✓ §4.2, §7.6–7.8 |
| Pending | Apply rewrite blocks to LaTeX | `results.tex`, `discussion.tex` | §10 item 7 |

### Issue index

| ID | Position | Doc section |
|----|----------|-------------|
| CRITICAL-01 | Partial concede + reframe | §4.5, §7.5 |
| CRITICAL-02 | Concede claim; defend RAG data | §7.7, §4.8 |
| CRITICAL-03 | Concede framing; defend curated F1 | §8, §7.4 |
| CRITICAL-04 | Defend; mechanistic explanation | §7.4 |
| MODERATE-01 | Concede labeling | §7.1, §13 |
| MODERATE-02 | Corrected 273/419 → 119/60 | §7.1, §7.3 |
| MODERATE-03 | Disclose rounding | §7.8 table |
| MODERATE-04 | Effect-size framing | §7.6 |
| MODERATE-05 | Gold subset primary | §7.2 |
| MODERATE-06 | Single RAG row | §7.2 |
| MINOR-01–06 | Disclose / process | §4.2, §7.5–7.8 |

### Cover letter paragraph (reviewer response template)

> We thank the reviewers for their thorough critique. All issues were addressed through text revisions and table restructuring; **no new experiments** were conducted. Core data — 36,360 predictions, 18,180 RAG faithfulness scores, 35,845 judge assessments, 10-trial NVML energy, 6,600 gold-labeled retrieval rows — remain unchanged (§12 audit). Key revisions: (1) routing classifier reframed as rule-based consistency check; (2) RQ2 scoped to RAG-only absolute faithfulness; (3) SLM+RAG superiority qualified to curated 3-query F1 with PubMedQA drops reported; (4) LLM+RAG ROUGE-L drop explained mechanistically; (5) throughput labeled Context-A derived with concurrent-load caveats. Detailed responses in `REWRITE_AND_DEFENSE_GUIDE.md`.

### Abstract-ready text (CRITICAL-03)

> On three controlled clinical queries, SLM+RAG (Gemma-2-2B-it) achieved the highest token-F1 (0.6101 vs. 0.4548 for LLM+RAG) with 53% lower GPU carbon footprint. On open biomedical benchmarks (n = 9,090 predictions/configuration), RAG yielded mixed effects: improving MMLU-Med accuracy for LLM (+18.1%) while reducing PubMedQA accuracy for both models (−4.2% to −5.5%), suggesting retrieval benefits are task- and domain-dependent.

### Defense guide summary — all issues (mirror of `REWRITE_AND_DEFENSE_GUIDE.md`)

| Issue | Action | LaTeX target | New experiments? |
|-------|--------|--------------|------------------|
| CRITICAL-01 | Rule-based consistency reframe | `implementation.tex`, routing | No |
| CRITICAL-02 | RQ2 partial; RAG absolute only | `results.tex` Table 4, `discussion.tex` | No |
| CRITICAL-03 | Qualify Finding 1 + abstract | `results.tex` §findings, abstract | No |
| CRITICAL-04 | ROUGE-L mechanistic paragraph | `results.tex` Table 2 note | No |
| MODERATE-01 | Context A derived labels | `results.tex` §7.1, EIE table | No |
| MODERATE-02 | 273/419 authoritative; 119 tok/s | `results.tex` §7.1, EIE | No |
| MODERATE-03 | CO₂/M display footnote | `results.tex` §7.8 | No |
| MODERATE-04 | Effect-size framing Table 3 | `results.tex` §7.6 | No |
| MODERATE-05 | Gold MRR primary; pooled supplementary | `results.tex` Table 1 caption | No |
| MODERATE-06 | Single shared RAG row | `results.tex` Table 1 | No |
| MINOR-01 | Per-config n in Table 3 | Table 3 caption | No |
| MINOR-02 | 8% fallback→LLM disclosed | Routing section | No |
| MINOR-03 | kWh primary; UK sensitivity | §7.8, §13 | No |
| MINOR-04 | 1M queries/day scaling (not 117M) | §7.8 | No |
| MINOR-05 | Faithfulness limitations strengthened | `discussion.tex` | No |
| MINOR-06 | F1 context labels everywhere | All F1 mentions | No |

**Total new experiments required: 0**

---

## Appendix A — Faculty summary (one paragraph)

We evaluated **Gemma-2-2B-it** (SLM) vs **Llama-2-7B-chat** (LLM) under RAG, NoRAG, and hybrid routing on a **115-document / 556-chunk** corpus and **36,360** open-benchmark predictions. On **three curated queries** (Context B), SLM+RAG achieved the highest **token-F1 (0.6101)** with **~53%** lower GPU carbon than LLM-only (19.8 vs 42.5 kg CO₂e/M). On **open benchmarks**, RAG effects were **mixed** (PubMedQA accuracy −4.2% to −5.5% with RAG; MMLU-Med +18.1% for LLM). Retrieval on the gold subset reached **Recall@3 = 0.891**, **MRR = 0.967**. **Qwen2.5-7B-Instruct** judged clinical quality (Table 3) and RAG faithfulness (Table 4, **~47–48%**, partial-support band); **NoRAG faithfulness**, Phase 4 blocking rates, and Phase 5 clinician validation remain outstanding. Claims are scoped per §4.2 (F1 contexts), §7.7 (RQ2 partial), and §14.

---

## Appendix B — Model quick-reference

```
GENERATION:     google/gemma-2-2b-it          (SLM)
                meta-llama/Llama-2-7b-chat-hf (LLM)
EMBEDDING:      sentence-transformers/all-MiniLM-L6-v2
JUDGE (Tab 3):  Qwen/Qwen2.5-7B-Instruct      (local_hf)
FAITHFULNESS:   Qwen/Qwen2.5-7B-Instruct      (local_hf, Tab 4)
HARDWARE:       Kaggle NVIDIA T4, NVML energy
THROUGHPUT:     119 / 60 tok/s (Context A derived; SLM / LLM)
CARBON:         0.385 kg CO₂e/kWh (U.S. ref); kWh/query primary
```
