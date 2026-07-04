# **Methodologicalical Implementation Plan**

---

## **Phase 1: System Architecture and Model Selection**

**Step 1.1: Model Selection and Environment Setup**

**Objective:**  
 Deploy a clinically aligned Small Language Model (SLM) optimized for retrieval-augmented generation (RAG), low-resource inference, and on-premises governance.

**Approach:**

**Model Selection:**

* Primary: Meerkat-7B (\~65–72% MedQA verified on USMLE split)  
* Fallback: Llama-3-Med-8B (stable reasoning, strong medical alignment)  
* Alternative: Meditron-7B or MedAlpaca-7B (if licensing or infrastructure constraints apply)

**Notes:**

* Only publicly licensed or institutionally approved clinical datasets will be used for fine-tuning or RAG.  
* Model performance claims are based on empirical evaluation with the internal MedQA test split.

**Quantization:**

* Apply 4-bit AWQ quantization to reduce VRAM to \~5–6 GB.  
* Empirical validation metrics:

  * MedQA accuracy change  
  * Explanation BLEU / ROUGE metrics  
  * Embedding cosine similarity

* Target thresholds: \<5% loss for generation, \<3% for embeddings.

**Hardware Deployment:**

* NVIDIA A10G (24 GB) or equivalent with vLLM for continuous batching and paged attention.  
* Latency: Expected 1.8–3.5 s per query under typical RAG+verification load (to be empirically measured).

**Deliverables:**

* Dockerized vLLM deployment  
* Streaming-capable REST API  
* Baseline MedQA evaluation (quantized vs. non-quantized)

**Validation Rationale:**

* Domain-aligned SLM reduces hallucinations and improves clinical term accuracy.  
* Quantization ensures low-resource feasibility without sacrificing reliability.

---

## **Phase 2: Knowledge Base Construction**

**Step 2.1: Evidence-Tiered Data Curation and Ingestion**

**Objective:**  
 Construct a structured, tiered clinical knowledge repository enabling safe, explainable retrieval.

**Evidence Tiers:**

* Tier 1 (High-evidence, immutable): NICE, FDA labels, PubMed RCTs, systematic reviews.  
* Tier 2 (Local protocols): Hospital SOPs, formularies, care pathways.  
* Tier 3 (Contextual but non-authoritative): Open-access textbooks, educational articles.

**ETL Pipeline:**

* Extract: PDFs, DOCX, HTML via Unstructured.io \+ OCR  
* Transform: clean text, normalize sections, fix formatting errors  
* Load: structured JSONL with standardized metadata

**Metadata Fields:**

* `source_id, evidence_level, document_type, publication_date, version_number, local_vs_global, section_hierarchy`

* Tier 1 sources updated monthly for critical updates; quarterly for others

**Deliverables:**

* Complete JSONL medical corpus  
* Evidence-tier catalog  
* ETL logs and parsing error reports

**Validation Rationale:**

* Evidence-tiering prevents low-authority documents from influencing clinical reasoning.  
* Legal compliance: only licensed or open-access content ingested.

---

**Step 2.2: Section-Aware and Fixed-Window Hybrid Chunking**

**Objective:**  
 Maximize retrieval performance while preserving clinical meaning and document logic.

**Approach:**

* Fixed-window: 400–500 tokens, 20–30% overlap  
* Section-aware:

  * Regex detection for headers (Dosage, Warnings, Contraindications)  
  * ML-based segmentation for noisy PDFs  
  * Parent-child hierarchy maintained

**Context Expansion:**

* Retrieved child chunk automatically loads full parent section to preserve warnings and logical context.

**Success Criterion:**

* Section-aware chunking must show statistically significant Recall@5 improvement over fixed-window (p \< 0.05).  
* Otherwise, default to fixed-window only.

**Deliverables:**

* Vector store (Milvus/Qdrant) with hierarchical links  
* Retrieval failure analysis  
* A/B evaluation report (Recall@5, Recall@10, statistical validation)

**Validation Rationale:**

* Ensures empirical selection between semantic and fixed-window chunking to avoid over-engineering.

---

## **Phase 3: Retrieval-Augmented Generation Pipeline**

**Step 3.1: Hybrid Retrieval Engine**

**Objective:**  
 Enable reliable retrieval using complementary dense and sparse methods.

**Approach:**

* Dense Retrieval: Evaluate BGE-Med vs. MedCPT on internal corpus  
  * Metrics: Recall@5, MAP@10  
  * Select the model with the highest internal performance

* Sparse Retrieval (BM25): Capture exact drug names, dosages, ICD codes, and abbreviations  
* Reciprocal Rank Fusion: Combine dense \+ sparse candidate scores  
* Cross-Encoder Reranking: MiniLM / Med-MiniLM refines top 10–15 chunks

**Deliverables:**

* Fully functioning hybrid retrieval microservice  
* Retrieval metrics (Recall@5, Recall@10)  
* Latency profile under concurrent queries

**Validation Rationale:**

* Reduces hallucinations by ensuring essential evidence is present in context.

* Embedding model internally validated to avoid misalignment.

---

## **Phase 4: Safety, Verification, and Controlled Generation**

**Step 4.1: Multi-Layer Verification (NLI \+ Reasoning)**

**Objective:**  
 Ensure factual correctness, clinical safety, and evidence grounding before release.

**Approach:**

1. Draft Generation: SLM produces evidence-linked answer with citation markers  
2. Sentence-Level Fact Verification: ClinicalNLI / DeBERTa-v3-med-NLI  
   * Outcome: Entailed / Contradicted / Neutral

3. Reasoning Verification: Lightweight reasoning verifier from a **different model family**  
   * Detect multi-hop inference errors

4. Safety Enforcement:  
   * Contradiction/unsupported claim → replaced with “Evidence insufficient.”  
   * Log event for audit  
   * Optional structured evidence summary returned

**Deliverables:**

* Verification pipeline  
* Hallucination interception logs  
* Safety audit logs

**Validation Rationale:**

* NLI catches factual errors; reasoning verifier catches inferential errors  
* Cross-model diversity prevents confirmation bias

---

## **Phase 5: Evaluation, Governance, and Human Oversight**

**Step 5.1: Benchmarking and Clinical Review**

**Objective:**  
 Assess accuracy, trustworthiness, safety, and explainability using automated benchmarks and expert review.

**Approach:**

* Automated Benchmarks: MedQA (USMLE), PubMedQA, MedMCQA  
* Targets for 7B–8B SLM:  
  * ≥60% MedQA baseline  
  * ≥65% stretch goal  
  * High faithfulness, evidence grounding

* Expanded Trustworthiness Metrics: F1-Score, Faithfulness, Context Recall  
* Refusal Testing: Adversarial queries, harmful requests  
  * True-Positive Refusal Rate: \>95%  
  * False-Positive Refusal Rate: \<5%

* Clinical Red Teaming: 500–1,000 clinician prompts rated Safe/Harmful/Incomplete  
* Traceability: Each output includes retrieved evidence, NLI & reasoning decisions, citation mapping, optional reasoning summary

**Deliverables:**

* Evaluation report  
* Clinically validated risk matrix (DTAC/NICE compliant)  
* Governance documentation

**Validation Rationale:**

* Safety, explainability, and accountability prioritized over raw accuracy

---

**Step 5.2: Energy, Environmental, and Cost Evaluation**

**Objective:**  
 Quantify sustainability and operational efficiency versus cloud LLM alternatives.

**Approach:**

* Measure energy per query using hardware monitoring (Wh, kWh)  
* Estimate CO₂ emissions using grid carbon intensity  
* Compare with GPT-4-class API under standardized token usage and batch loads  
* Compute 5-year TCO: hardware, storage, cooling, maintenance

**Deliverables:**

* Sustainability and operational efficiency report  
* Defensible comparison with cloud alternatives

---

**Step 5.3: Regulatory Compliance Assessment**

**Objective:**  
 Ensure alignment with healthcare governance and data-sovereignty frameworks.

**Approach:**

* HIPAA/GDPR/NHS DTAC compliance matrix  
* Compare on-prem SLM+RAG vs. cloud-based LLMs  
* Produce clinical safety case documenting: risk controls, audit logging, data-handling, decision traceability

**Deliverables:**

* Compliance matrix  
* Safety case report  
* Audit-ready governance documentation

---

