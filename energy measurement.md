## **1\. Measurement goal and basic idea**

**Environment:** This guide is for **Kaggle notebook**. Enable GPU in Settings (e.g. T4 x2). Install `pynvml` in the first cell; NVML works on Kaggle’s NVIDIA GPU.

You want, for each configuration:

* SLM (No RAG)  
* SLM \+ RAG  
* LLM (No RAG)  
* LLM \+ RAG  
* Routing (Hybrid)

to measure:

* **Total GPU energy for a batch of queries** (J or kWh).  
* **Energy per query** (kWh/query).  
* Then convert to **CO₂e** using a grid factor (you used 0.385 kg CO₂e/kWh in the current draft).\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​

We will use **NVIDIA’s NVML API** via `pynvml` to read **total energy consumption** at the start and end of an inference batch, which is simpler and more accurate than your current “token-count heuristic”.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​\[[ml](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)\]​

---

## **2\. Environment setup**

## **2.1. Install dependencies**

In your **Kaggle notebook** (first cell):

```bash
pip install pynvml torch transformers
```

NVML is part of the NVIDIA driver; `pynvml` is just a Python wrapper. On Kaggle, enable **GPU** in the right-hand **Settings** so NVML can read GPU energy.\[[ml](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)\]​

## **2.2. Initialize NVML in Python**

In your main measurement script:

python  
`import pynvml`  
`import time`  
`import torch`

`pynvml.nvmlInit()`  
`handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0`

* `nvmlDeviceGetTotalEnergyConsumption(handle)` returns **total energy in millijoules** consumed by that GPU since boot (on supported GPUs).\[[ml](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)\]​

---

## **3\. Define a generic energy measurement wrapper**

You already have code to run a batch of queries for a given config. We’ll wrap that in an energy measurement function.

## **3.1. Pseudocode for measurement**

python  
`def run_inference_batch(config_name, model, rag_enabled, questions):`  
    `"""`  
    `Your existing function that:`  
      `- runs model inference on a list of queries`  
      `- uses RAG or not depending on rag_enabled`  
      `- returns model outputs`  
    `"""`  
    `# TODO: plug your existing routing/RAG pipeline here`  
    `pass`

`def measure_energy_for_config(config_name, model, rag_enabled, questions, repeats=3):`  
    `results = []`

    `for r in range(repeats):`  
        `# Optional: warm-up run (to avoid first-run overhead)`  
        `_ = run_inference_batch(config_name, model, rag_enabled, questions[:5])`  
        `torch.cuda.synchronize()`

        `# Record start time and energy`  
        `start_time = time.time()`  
        `start_energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)`

        `# Run the actual batch`  
        `outputs = run_inference_batch(config_name, model, rag_enabled, questions)`  
        `torch.cuda.synchronize()  # ensure all GPU work is finished`

        `# Record end time and energy`  
        `end_time = time.time()`  
        `end_energy_mJ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)`

        `elapsed_s = end_time - start_time`  
        `consumed_mJ = end_energy_mJ - start_energy_mJ  # millijoules`  
        `consumed_J = consumed_mJ / 1000.0              # joules`  
        `consumed_kWh = consumed_J / 3_600_000.0        # 1 kWh = 3.6e6 J`

        `num_queries = len(questions)`  
        `kWh_per_query = consumed_kWh / num_queries`

        `results.append({`  
            `"run": r,`  
            `"elapsed_s": elapsed_s,`  
            `"consumed_kWh": consumed_kWh,`  
            `"kWh_per_query": kWh_per_query,`  
        `})`

    `return results`

**Key points:**

* We measure **total GPU energy** over a full batch of N queries, then divide by N.  
* We repeat a few times (`repeats=3`) to get a mean and standard deviation, as the reviewer wants more reliable numbers.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​\[[ml](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)\]​

---

## **4\. Running this for each configuration**

## **4.1. Prepare a fixed query set**

Use a **fixed list of N queries** for energy tests so comparisons are fair:

python  
`questions = load_energy_test_queries()  # e.g., your 12 extended clinical queries, or 50 queries`

They should be representative of your typical workload (mix of simple and complex clinical questions).\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​

## **4.2. Run measurements**

Example of how to run:

python  
`all_results = {}`

*`# 1. SLM (No RAG)`*  
`slm_model = load_slm_model()  # Gemma-2B`  
`all_results["SLM_NoRAG"] = measure_energy_for_config(`  
    `"SLM_NoRAG",`  
    `slm_model,`  
    `rag_enabled=False,`  
    `questions=questions,`  
    `repeats=3`  
`)`

*`# 2. SLM + RAG`*  
`all_results["SLM_RAG"] = measure_energy_for_config(`  
    `"SLM_RAG",`  
    `slm_model,`  
    `rag_enabled=True,`  
    `questions=questions,`  
    `repeats=3`  
`)`

*`# 3. LLM (No RAG)`*  
`llm_model = load_llm_model()  # Llama-2-7B or newer`  
`all_results["LLM_NoRAG"] = measure_energy_for_config(`  
    `"LLM_NoRAG",`  
    `llm_model,`  
    `rag_enabled=False,`  
    `questions=questions,`  
    `repeats=3`  
`)`

*`# 4. LLM + RAG`*  
`all_results["LLM_RAG"] = measure_energy_for_config(`  
    `"LLM_RAG",`  
    `llm_model,`  
    `rag_enabled=True,`  
    `questions=questions,`  
    `repeats=3`  
`)`

*`# 5. Routing (Hybrid)`*  
*`# Here your run_inference_batch should implement routing:`*  
*`#   - simple queries → SLM (+/- RAG)`*  
*`#   - complex queries → LLM (+/- RAG)`*  
`all_results["Routing_Hybrid"] = measure_energy_for_config(`  
    `"Routing_Hybrid",`  
    `{"slm": slm_model, "llm": llm_model},  # or a pipeline object`  
    `rag_enabled=True,  # or pass routing config separately`  
    `questions=questions,`  
    `repeats=3`  
`)`

You’ll adapt `run_inference_batch` so that:

* For **routing**, it calls your existing routing classifier to choose SLM vs LLM per query, and uses RAG accordingly.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​

---

## **5\. Compute summary statistics and CO₂e**

## **5.1. Aggregate results per configuration**

After all runs:

python  
`import statistics`

`def summarize_energy(results):`  
    `kWh = [r["consumed_kWh"] for r in results]`  
    `per_q = [r["kWh_per_query"] for r in results]`  
    `return {`  
        `"kWh_mean": statistics.mean(kWh),`  
        `"kWh_std": statistics.pstdev(kWh),`  
        `"kWh_per_query_mean": statistics.mean(per_q),`  
        `"kWh_per_query_std": statistics.pstdev(per_q),`  
    `}`

`summary = {name: summarize_energy(res) for name, res in all_results.items()}`

## **5.2. Convert to CO₂e**

Use the grid factor you already cite: **0.385 kg CO₂e/kWh**.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/fbd77cc4-d708-451c-a818-41f999e657e4/Preparation_of_Papers_for_IEEE_ACCESS__17___1_-1.pdf)\]​

python  
`GRID_FACTOR = 0.385  # kg CO2e per kWh`

`for name, s in summary.items():`  
    `kWh_per_query = s["kWh_per_query_mean"]`  
    `co2_per_query_kg = kWh_per_query * GRID_FACTOR`  
    `co2_per_million_queries_kg = co2_per_query_kg * 1_000_000`

    `s["co2_per_query_kg"] = co2_per_query_kg`  
    `s["co2_per_million_queries_kg"] = co2_per_million_queries_kg`

Now you can build a clean table like:

| Strategy | kWh/query (mean ± std) | kg CO₂e/query | kg CO₂e / 1M queries |
| ----- | ----- | ----- | ----- |
| SLM Only | … | … | … |
| SLM \+ RAG | … | … | … |
| LLM Only | … | … | … |
| LLM \+ RAG | … | … | … |
| Routing (Hybrid) | … | … | … |

This directly replaces your old heuristic numbers (0.008 vs 0.0013 kWh/query, 7.05 kg CO₂e per million queries, etc.), which the reviewer flagged as inconsistent.Review-2-2.pdf+1

---

## **6\. How to describe this in the paper**

In **Methods – Energy and Carbon Accounting**:

* Say you used **pynvml** / NVML’s `nvmlDeviceGetTotalEnergyConsumption` to measure **total GPU energy** over batches of N queries, repeated 3 times per configuration, and report mean ± standard deviation.\[[ml](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)\]​\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​  
* Specify:  
  * GPU type (e.g., NVIDIA A100).  
  * Batch size / number of queries per run.  
  * Grid factor (0.385 kg CO₂e/kWh) and source.  
* Clearly state limitations:  
  * Does not include CPU energy or data center overhead.  
  * Single‑GPU, single‑node measurement; numbers approximate but internally consistent.

In **Results – Environmental Impact**:

* Replace the previous small and inconsistent numbers with:  
  * Per‑query kWh for each configuration.  
  * Relative % reduction vs LLM‑only baseline.  
  * CO₂e per million queries (and extrapolated annual savings if you still want that, but now computed consistently).

This directly answers reviewer Q2 (“Please provide hardware-level power measurements or standardized tooling results with uncertainty estimates”) and fixes the internal contradictions they pointed out.\[[ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/87117853/2ce4d7d4-0f3f-4bc9-8f5a-218245575f24/Review-2-2.pdf)\]​

---

If you want, next I can:

* Draft exact LaTeX for the new energy/carbon table, or  
* Give you a minimal, ready‑to‑run Colab cell that plugs into your existing inference code.

