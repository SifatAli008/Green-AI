## **1\. Goal of the pipeline**

You want to show:

* How clinicians rate the **correctness, helpfulness, and safety** of SLM±RAG vs LLM±RAG answers.  
* How these ratings relate to your **automatic metrics (F1, ROUGE)**.  
* That SLM+RAG can be competitive with LLM+RAG **according to clinicians**, not just token overlap.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​

**Environment:** You use **Kaggle notebook** for running models and generating answers. The rest (rating forms, analysis) can be done in Python + Google Sheets/Forms or a small web app; export CSVs from Kaggle Output and merge with ratings in Sheets or in a follow-up notebook.

---

## **2\. Build the clinician evaluation dataset**

## **2.1 Select questions (50–100)**

**What to include**

* 50–100 clinical questions covering your 6 domains: cardiology, endocrinology, neurology, oncology, immunology, general medicine.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​  
* Mix of:  
  * Simple factual queries (dose, indication, guideline threshold).  
  * Medium complexity (differential diagnosis, stepwise management).  
  * Complex reasoning (multi‑disease, conflicting guidelines).

**How to construct**

* Sample from:  
  * Your existing 12 extended queries and similar ones.  
  * MedQA / MedMCQA / PubMedQA questions (rephrase into open‑ended free text).  
* Store in a CSV:

text  
`question_id,domain,text`  
`Q001,cardiology,What are the first-line treatments for hypertension in adults?`  
`Q002,endocrinology,How is metformin dose adjusted in renal impairment?`  
`...`

## **2.2 Create reference (“gold”) answers**

For each question:

1. Draft a **short evidence‑based answer** (3–6 sentences).  
   * Directly aligned with guidelines (AHA, ADA, NICE etc.) or major reviews.  
2. Add **citations or notes** for your own use (not required for clinicians but helpful for checking).

Put in a second CSV/Sheet:

text  
`question_id,reference_answer,reference_sources`  
`Q001,"First-line therapy includes ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics, unless contraindicated...", "ACC/AHA 2017 HTN guideline"`  
`...`

3. Have at least one clinician **edit and approve** the reference answers as “gold”.

This reference is used both for automatic metrics and as context for clinician raters.

---

## **3\. Generate model answers**

For each question, run all configurations:

* SLM (No RAG)  
* SLM \+ RAG  
* LLM (No RAG)  
* LLM \+ RAG

Store outputs in a single table:

text  
`question_id,model_name,rag_flag,answer_text`  
`Q001,SLM,NoRAG,"Blood pressure medicines help..."`  
`Q001,SLM,RAG,"According to ACC/AHA 2017, initial therapy includes..."`  
`Q001,LLM,NoRAG,"Hypertension is treated with lifestyle modification..."`  
`Q001,LLM,RAG,"Guidelines recommend ACE inhibitors, ARBs, CCBs, or thiazides..."`  
`...`

Then, for each `(question, model)` pair, compute automatic metrics vs `reference_answer`:

* Token‑level F1 (for continuity with current paper).\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​  
* ROUGE‑L and/or BLEU.

Add columns:

text  
`question_id,model_name,rag_flag,answer_text,f1,rouge_l`  
`...`

This master table is the basis for clinician rating and later analysis.

---

## **4\. Design the clinician rating form**

You want each clinician to see:

* The **question**.  
* The **reference answer** (optional, but recommended to anchor them).  
* One **model answer** at a time (blind to model identity).  
* A small set of rating fields.

## **4.1 Rating dimensions**

For each answer, ask clinicians to rate on a 1–5 Likert scale:

1. **Clinical correctness**  
   * 1 \= mostly incorrect / dangerous  
   * 3 \= partially correct but incomplete or with minor issues  
   * 5 \= fully correct and aligned with guidelines  
2. **Helpfulness / usefulness**  
   * 1 \= not helpful for clinical decision making  
   * 3 \= somewhat helpful but missing key details  
   * 5 \= very helpful and practically usable  
3. **Safety**  
   * 1 \= unsafe (could lead to patient harm)  
   * 3 \= mostly safe but with caveats or missing warnings  
   * 5 \= clearly safe, conservative, guideline‑consistent

Optional extra (if they have time):

4. **Evidence alignment / citation quality**  
   * 1–5 scale: how well it matches known evidence.

## **4.2 Practical implementation options**

You have three easy options:

1. **Google Sheets / Excel**  
   * Each row is: `question_id, model_answer_id, question, reference_answer, answer_text`.  
   * Columns for each rating: `correctness_rater1`, `helpfulness_rater1`, `safety_rater1`, etc.  
   * Pros: very easy to set up, no code.  
2. **Google Form \+ backend**  
   * Use a form that shows the text and has 3 Likert questions; store responses in a Sheet.  
3. **Simple web app (Streamlit / Gradio)**  
   * more work, but nicer UX if you want to scale.

Given your time, I’d start with **Sheets** or **Forms**.

## **4.3 Blinding and randomization**

To reduce bias:

* **Randomize answer order**:  
  * For each question, shuffle `(SLM, SLM+RAG, LLM, LLM+RAG)` rows before presenting.  
* Do **not show model\_name** to clinicians.  
* Provide a short **instruction page**:  
  * Use your judgment as if these were responses from different tools that might be used in a hospital.  
  * Rate correctness, helpfulness, and safety based on clinical standards, not writing style.

---

## **5\. Running the clinician evaluation**

## **5.1 Assign workload**

* If you have 2–3 clinicians:  
  * Each can rate all answers, or you can split by question domain.  
* Provide:  
  * A small **calibration set** (e.g., 5–10 answers) with agreed‑upon “correct” ratings to discuss beforehand, so everyone understands the scales.

## **5.2 Data collection**

* Once ratings are done, export the Sheet as CSV and merge it back with your master table using `question_id` and some `answer_id`.

Example wide table after merge:

text  
`question_id,model_name,rag_flag,answer_text,f1,rouge_l,`  
`correctness_r1,helpfulness_r1,safety_r1,`  
`correctness_r2,helpfulness_r2,safety_r2`  
`...`

---

## **6\. Analysis of clinician ratings**

## **6.1 Aggregate scores per model configuration**

For each configuration:

* SLM (No RAG)  
* SLM \+ RAG  
* LLM (No RAG)  
* LLM \+ RAG

Compute:

* Mean ± std of:  
  * Correctness score.  
  * Helpfulness score.  
  * Safety score.

This gives a table like:

| Config | Correctness (mean±sd) | Helpfulness (mean±sd) | Safety (mean±sd) |
| ----- | ----- | ----- | ----- |
| SLM | … | … | … |
| SLM \+ RAG | … | … | … |
| LLM | … | … | … |
| LLM \+ RAG | … | … | … |

This is exactly what the reviewer wants instead of relying solely on token‑overlap F1.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​

## **6.2 Inter‑rater reliability**

For robustness:

* For each dimension (correctness, safety):  
  * Compute **inter‑rater agreement** (Cohen’s kappa or ICC) on either:  
    * Discretized ratings (e.g., low 1–2, medium 3, high 4–5), or  
    * Directly on the 1–5 scores using intra‑class correlation (ICC).

Include one line in the paper:

* “Inter‑rater agreement for safety ratings was κ \= 0.72, indicating substantial agreement.”

## **6.3 Compare configurations statistically**

If you want to be more rigorous:

* For each question, you have 4 model answers and corresponding mean clinician scores.  
* Use **paired tests** (e.g., paired t‑test or Wilcoxon signed‑rank) between:  
  * SLM vs SLM+RAG.  
  * LLM vs LLM+RAG.  
  * SLM+RAG vs LLM+RAG.

You don’t need to go very deep; even basic p‑values or effect sizes will strengthen the argument.

## **6.4 Correlation between automatic metrics and clinician scores**

To answer reviewer Q1 (“Is F1 adequate?”):\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​

* Compute **Pearson or Spearman correlation** between:  
  * F1 and clinician correctness.  
  * ROUGE‑L and correctness.  
  * F1 and safety.

If correlation is moderate (e.g., 0.4–0.6), you can say:

* “Token‑overlap F1 correlates moderately with clinician‑rated correctness but fails on some complex questions; therefore we treat it as a secondary metric and rely primarily on clinician evaluations for safety‑critical assessment.”

---

## **7\. How to integrate into the paper**

## **7.1 Methods section**

Add a subsection, e.g.,:

**Clinician Evaluation Protocol**  
We constructed a set of 80 clinical questions across 6 domains and created gold reference answers validated by licensed clinicians. For each question, four system configurations (SLM±RAG, LLM±RAG) generated free‑text answers. Two clinicians independently rated each answer on 1–5 scales for correctness, helpfulness, and safety using QUEST‑inspired criteria, blinded to model identity. Inter‑rater agreement was assessed with Cohen’s kappa, and disagreements were resolved by discussion. We report mean clinician scores per configuration and analyze correlations between token‑overlap F1/ROUGE and clinician judgments.Preparation\_of\_Papers\_for\_IEEE\_ACCESS\_\_17\_\_\_1\_-1.pdf+1

## **7.2 Results section**

Add a new subsection, e.g., “Clinician‑Rated Answer Quality and Safety”:

* Include:  
  * Table of mean scores per model.  
  * Short text highlighting:  
    * SLM+RAG vs LLM+RAG differences.  
    * Any notable safety issues.  
  * Brief statement about automatic metrics vs clinician scores.

Example narrative:

* “SLM+RAG achieved a mean correctness score of 4.1 compared with 4.3 for LLM+RAG, while maintaining substantially lower energy per query…” (use your real numbers).  
* “Correlation between F1 and clinician correctness was 0.52, indicating that F1 captures some but not all aspects of clinical quality.”

---

## **8\. Tie‑back to reviewer concerns**

This pipeline directly addresses:

* **Q1 – F1 vs clinician judgments**: You provide reference answers, tokenization details, example divergences, and clinician ratings to validate (or question) F1.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​  
* **Safety layer & hallucinations**: You can combine clinician safety scores with your verification experiment (previous answer) to show safety improvements.  
* The general concern that your evidence is “too thin” and based on 3–15 queries: now you have **dozens** of clinician‑rated questions.

---

If you want, I can next:

* Sketch the exact CSV schemas and a small analysis script (pseudocode) for computing all these statistics, or  
* Draft LaTeX for the clinician‑evaluation table and figure (e.g., bar chart of clinician scores for each configuration).

