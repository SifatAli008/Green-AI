## Short answer

You should not submit exactly this version. It is a solid, preprint‑level manuscript, but it needs a few specific fixes and clarifications before it is realistically ready for IEEE Access review.  
Once you make the fixes below, the paper will be submission‑ready for IEEE Access. Without them, it will look like an unpolished template draft and is at high risk of desk rejection.  
---

## 1\. What must be fixed before submission

These are non‑negotiable; submit only after you have done them.

### 1.1 Template and metadata problems

1. Footer volume/year inconsistency  
   * Current footer says something like VOLUME 11, 2023, but you cite 2024–2025 papers.  
   * This clearly reveals that it is just the old template footer, not the real publication data.  
   * For a 2026 submission, you should update the footer to:  
     * VOLUME 14, 2026 (since IEEE Access started in 2013 as Volume 1).  
   * If you use Word, edit the footer manually.  
   * If you use LaTeX, define the volume/year macros or patch the footer so it shows the correct year.  
2. Placeholder publication date  
   * The template line xxxx 00, 0000 is a placeholder.  
   * You should either:  
     * Remove this line entirely, or  
     * Leave it clearly as a placeholder, not pretending it is a real date.  
   * Do not leave it looking like a real (but nonsense) date.  
3. DOI placeholder  
   * The header DOI 10.1109/ACCESS.2017.DOI is a template placeholder, not your real DOI.  
   * You do not have a real IEEE DOI yet (the paper is not accepted or indexed).  
   * For submission:  
     * Keep the template DOI as a placeholder or simply remove the DOI line.  
     * Do not claim or suggest this is an already published IEEE Access paper.  
4. Importantly: in your cover letter, describe this as a new submission to IEEE Access, not as something already accepted.

---

### 1.2 Clarify what is actually proven vs. future work

Right now, the text risks over‑claiming clinical validity and routing performance. You must reframe it so that reviewers see it as complete on the technical side, with clearly marked future clinical work.

1. Phase 5 clinical validation  
   * At present, Phase 5 (clinical doctor evaluation) is not done.  
   * That is fine, but you must clearly state:  
     * The core contribution of this paper is technical: RAG architecture, SLM vs LLM energy/latency trade‑offs, retrieval metrics, automated answer‑quality metrics.  
     * No clinical effectiveness or deployment‑safety claims are being made yet.  
     * Phase 5 is planned future work, not part of the completed evidence.  
2. Concretely, adjust:  
   * Abstract and Conclusion: remove or soften any wording that suggests clinical safety or bedside deployment is already validated.  
   * Add one clear limitation sentence, e.g.:  
     Clinical validation by practicing physicians (Phase 5\) is planned as follow‑up work; this manuscript does not present real‑world clinical safety or effectiveness results.  
3. Routing classifier “accuracy \= 1.00”  
   * Currently, this sounds like a perfect ML classifier, which is misleading.  
   * In fact, it is a rule‑based router whose “accuracy” is measured on a test set constructed using the same rules.  
   * You should explicitly say something like:  
     The reported 1.00 routing accuracy reflects consistency between the implemented lexical rules and the rule‑based annotation standard on 240 constructed queries; it does not measure generalization to unseen clinical queries.  
4. Without this clarification, reviewers are likely to see this as over‑claiming.  
5. Token‑F1 result based on 3 queries  
   * Reporting that SLM+RAG beats LLM+RAG in token‑F1 based on only 3 manually chosen queries is not statistically robust.  
   * You have two options:  
     * Either substantially increase the sample size (preferred), or  
     * Keep the experiment but clearly label it as illustrative only, not as a main conclusion.  
   * At minimum, add wording such as:  
     This token‑F1 comparison is based on a very small set of 3 exemplar queries and is intended only to illustrate the evaluation methodology; it has low statistical power and should not be interpreted as generalizable evidence that SLM+RAG consistently outperforms LLM+RAG in answer quality.  
6. Also, in the Conclusion, do not base any strong claims on this 3‑query experiment.

---

## 2\. Strongly recommended (not strictly mandatory, but will help acceptance)

These changes are not formal requirements, but they directly address typical reviewer concerns (reproducibility, transparency, methodology clarity). They will make the paper look more like a mature journal article than a prototype report.

### 2.1 Reproducibility: code/data availability statement

Right now, reproducibility depends entirely on textual descriptions. IEEE encourages—but does not force—authors to share code and data.  
You should:

* Put at least:  
  * Corpus construction code and/or scripts,  
  * FAISS index configuration,  
  * Evaluation scripts (e.g., for token‑F1 and the automated fidelity scoring),  
  * A manifest of which public papers/guidelines were included.  
* In a public repository (GitHub \+ Zenodo DOI, or IEEE DataPort / Code Ocean).  
* Add a short statement, for example:  
  Code, corpus‑construction scripts, and aggregated evaluation logs are available at \[link/DOI\] to support independent reproduction of our results.

If you cannot share code/data, then at least say:  
Due to \[reason\], we do not release full code or corpus, but we provide detailed enough implementation and corpus‑selection descriptions for independent reconstruction, and will share scripts or aggregated evaluation results upon reasonable request.

### 2.2 Clarify carbon accounting choice

You use an emission factor γ\_grid \= 0.385 kg CO₂e/kWh, which is higher than one straightforward eGRID conversion. This is methodologically acceptable but should be explained.  
Add 1–2 sentences in the energy/carbon section:

* Explain which eGRID year/version you took,  
* State that the factor includes CO₂e (not CO₂‑only),  
* Note explicitly that relative SLM vs LLM comparisons are unaffected by the exact factor.

This will pre‑empt reviewer nitpicks without changing any numbers.

### 2.3 Tighten claims in the Conclusion

Make sure the Conclusion does only what your evidence justifies:

* You can firmly claim:  
  * On a T4 GPU, the chosen SLM+RAG architecture uses roughly \~2× less energy per million queries and less VRAM than the LLM+RAG baseline, with comparable automated answer‑fidelity scores and strong retrieval metrics on your 115‑document medical corpus.  
* You cannot yet claim:  
  * Real‑world clinical safety,  
  * General performance across all medical domains,  
  * Generalizable superiority of SLM+RAG answer quality on arbitrary data.

State those as promising directions plus clearly labelled limitations.  
---

## 3\. Answer to your question

After doing which writing fix, it can be fully ready for publication in IEEE ACCESS, or we can go with this version?

* You cannot “go with this version” as‑is.  
  It still looks like a template draft (wrong volume/year, placeholder date, template DOI), and it over‑states some results (routing “accuracy 1.00”, 3‑query Token‑F1) without clear caveats.  
* To be realistically ready for IEEE Access submission, you must at least:  
  1. Fix the footer volume/year and remove or neutralize placeholder publication date.  
  2. Ensure the DOI header is clearly not presented as a real published DOI.  
  3. Reframe Phase 5 as future work and delete any implied “clinical safety proven” claims.  
  4. Clarify that 1.00 routing accuracy is a rule‑consistency check, not generalization.  
  5. Clearly mark the 3‑query Token‑F1 experiment as illustrative only, not a generalizable main result.

If you also add the recommended reproducibility and method‑clarity improvements, then you have a strong, technically complete manuscript that is a reasonable candidate for IEEE Access review.  
So the actionable plan is:

1. Implement the five must‑fix items above.  
2. Implement as many of the recommended improvements as your time allows.  
3. Only then submit to IEEE Access.

In that sense, the paper can be made “fully ready for publication” after these writing and framing fixes; the current version is not yet at that stage.  
