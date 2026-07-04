## **1. Dataset Scope**

A total of **20–30 clinical queries** are selected, including:

* 3 core queries
* 12 extended queries
* Additional realistic clinical questions

The same query set is consistently used across all evaluations:

* **Recall@n (retrieval performance)**
* **BERTScore (answer quality)**
* **LLM-as-judge (clinical quality)**
* **DeepEval Faithfulness (hallucination/safety)**
* **Medical LLM comparison**

---

## **2. Corpus and Retrieval Performance (Recall@n)**

This section focuses solely on **retrieval effectiveness**, without introducing additional metrics such as F1.

* Corpus statistics are reported:

  * 115 documents
  * 556 chunks
  * 50–100 ms latency
  * ~100 MB index size

* For evaluation:

  * Each query is manually annotated with **3–5 relevant chunks**
  * Retrieval is evaluated using a **hybrid index (dense + sparse)**

* Metrics:

  * **Recall@3**
  * **Recall@5** (optionally Recall@10)

* Table:
  **Table 1 – Retrieval Performance**

  * Columns: `Recall@3`, `Recall@5` (mean ± std)
  * Rows: Hybrid (optionally Dense-only / Sparse-only / Hybrid)

**Key statement:**

> High Recall@3 indicates that the system consistently retrieves relevant evidence, which contributes to improved BERTScore and Faithfulness in subsequent stages.

---

## **3. Impact of RAG on Answer Quality (BERTScore)**

This section evaluates the effect of RAG on generated responses.

* Configurations:

  * SLM-NoRAG
  * SLM-RAG
  * LLM-NoRAG
  * LLM-RAG
  * Routing-Hybrid

* Metrics:

  * Existing **F1 score** (retained as-is)
  * **BERTScore (mean ± std)**
  * **Accuracy (mean ± std)** on MedQA, MMLU-Med, PubMedQA label accuracy

* Table:
  **Table 2 – Answer Quality**

  * Columns: `Configuration`, `F1`, `BERTScore`, benchmark accuracy (mean ± std)

**Key observation:**

> With the inclusion of RAG, SLM-RAG achieves BERTScore values comparable to LLM-RAG, particularly for simpler queries.

---

## **4. Clinical Quality via LLM-as-Judge**

This subsection introduces **model-based clinical evaluation**.

* Judge model: GPT-4-level or any practical medical LLM

* Evaluation rubric (0–5 scale):

  * Correctness
  * Completeness
  * Clinical relevance

* Process:

  * Input: question + reference answer + model output + rubric
  * Scores are assigned per query per configuration

* Table:
  **Table 3 – LLM-as-Judge Clinical Scores**

  * Columns: `Configuration`, `Correctness`, `Completeness`, `Clinical relevance` (mean ± std)

**Key statement:**

> Even when lexical overlap is limited, LLM-as-judge evaluation shows that SLM-RAG maintains high correctness and clinical relevance, supporting its practical usability.

---

## **5. Hallucination and Safety (DeepEval Faithfulness)**

This section evaluates **factual grounding and hallucination behavior**.

* Tool: DeepEval

* Metric:

  * **Faithfulness score** (evidence alignment)

* Optional derived metric:

  * **Hallucination rate = 1 − Faithfulness**

* Configurations:

  * SLM-NoRAG vs SLM-RAG
  * LLM-NoRAG vs LLM-RAG
  * Routing-Hybrid (optional)

* NoRAG grounding (comparable to RAG):

  * PubMedQA: task abstract in ``context``
  * MCQ (MedQA/MMLU): question stem + answer choices

* Table:
  **Table 4 – Faithfulness and Hallucination**

  * Columns: `Configuration`, `Faithfulness (%)`, `Hallucination rate (%)`

**Key statement:**

> Incorporating RAG improves Faithfulness and reduces hallucination rates, demonstrating the effectiveness of the evidence-grounded safety framework.

---

## **6. Comparison with Recent Medical LLMs**

This section benchmarks the system against recent medical language models.

### **Plan A (if models can be run)**

* Models:

  * Meditron
  * BioMistral
  * OpenMedLM

* Evaluation:

  * Same internal query set
  * **BERTScore against reference answers**

* Table:
  **Table S1 – Internal BERTScore Comparison**

  * Rows: `SLM-RAG (this work)`, `LLM-RAG`, baseline models
  * Columns: `BERTScore`

---

### **Plan B (literature-based comparison)**

* Models:

  * Med-PaLM 2
  * MedGemma
  * Meditron
  * BioMistral
  * OpenMedLM

* Metrics collected from published papers:

  * MedQA score
  * PubMedQA score

* Your system:

  * Internal **BERTScore / F1** reported separately

* Table:
  **Table S2 – Benchmark Comparison**

  * Columns: `MedQA`, `PubMedQA`, `Other`, `BERTScore (internal)`

**Key statement:**

> While benchmark scores are taken from prior work, the proposed SLM-RAG system demonstrates comparable performance within its internal evaluation setting.

---

## **7. Final Results Section Structure (Updated)**

Organize the Results section as follows:

1. **Corpus and Retrieval Performance**

   * Corpus statistics + Recall@3 / Recall@5

2. **Impact of RAG on Answer Quality**

   * Response length, latency, F1, and BERTScore

3. **Routing and Efficiency**

   * Routing F1, latency under load, energy/carbon metrics

4. **Clinical Quality via LLM-as-Judge**

   * Table 3 + concise analysis

5. **Hallucination and Safety Analysis**

   * Table 4 + safety discussion (score NoRAG with task-level grounding)

6. **Comparison with Recent Medical LLMs**

   * Table S1 or S2

