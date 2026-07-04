# =============================================================================
# MEASUREMENT CONFIG — Single source of truth for paper numbers
# Used to fix numerical inconsistencies (Revision Plan §1).
# Update this file when changing methodology; paper must cite these contexts.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. ENERGY (kWh per query) — NVML-based measurements
# -----------------------------------------------------------------------------
# All energy values now come from GPU telemetry (NVML) on Kaggle T4 Draft
# sessions, averaged over 10 runs of 20 queries each. These replace the old
# heuristic/token-based estimates and MUST be used consistently in the paper.
#
# Per-query energy (mean over runs) by strategy:
ENERGY_KWH_PER_QUERY_SLM_NORAG = 0.000052   # kWh/query (SLM_NoRAG)
ENERGY_KWH_PER_QUERY_SLM_RAG   = 0.000052   # kWh/query (SLM_RAG)
ENERGY_KWH_PER_QUERY_LLM_NORAG = 0.000111   # kWh/query (LLM_NoRAG)
ENERGY_KWH_PER_QUERY_LLM_RAG   = 0.000112   # kWh/query (LLM_RAG)
ENERGY_KWH_PER_QUERY_ROUTING   = 0.000101   # kWh/query (Routing_Hybrid)

# Convenience: kWh per million queries (multiply by 1e6).
ENERGY_KWH_PER_MILLION_SLM_NORAG = ENERGY_KWH_PER_QUERY_SLM_NORAG * 1e6
ENERGY_KWH_PER_MILLION_SLM_RAG   = ENERGY_KWH_PER_QUERY_SLM_RAG * 1e6
ENERGY_KWH_PER_MILLION_LLM_NORAG = ENERGY_KWH_PER_QUERY_LLM_NORAG * 1e6
ENERGY_KWH_PER_MILLION_LLM_RAG   = ENERGY_KWH_PER_QUERY_LLM_RAG * 1e6
ENERGY_KWH_PER_MILLION_ROUTING   = ENERGY_KWH_PER_QUERY_ROUTING * 1e6

# -----------------------------------------------------------------------------
# 2. LATENCY (seconds)
# -----------------------------------------------------------------------------
# CONTEXT A — Single-query baseline: one request, no concurrency.
SINGLE_QUERY_LATENCY_SLM_S = 2.3
SINGLE_QUERY_LATENCY_LLM_S = 7.0

# CONTEXT B — Mean per-request latency under concurrent load (load test).
# 5 users:  ~22.19–22.95 s;  10 users: ~49.55–51.49 s
LATENCY_5_USERS_MEAN_S = 22.19   # with RAG
LATENCY_10_USERS_MEAN_S = 49.55   # with RAG

# -----------------------------------------------------------------------------
# 3. MEMORY (GB)
# -----------------------------------------------------------------------------
# CONTEXT A — Model card / minimal footprint (single model, inference only).
MEMORY_SLM_GB = 4   # Gemma-2B typical
MEMORY_LLM_GB = 14  # Llama-2-7B typical

# CONTEXT B — Peak GPU memory under full pipeline (both models or full stack).
# From load test: torch.cuda.max_memory_allocated() ≈ 14.28–14.37 GB.
MEMORY_PEAK_LOAD_TEST_GB = 14.37

# -----------------------------------------------------------------------------
# 4. FAISS INDEX
# -----------------------------------------------------------------------------
# Index type: IndexFlatIP (inner product). Vectors are L2-normalized before
# indexing and before query; inner product of normalized vectors = cosine similarity.
# Do NOT report "L2 distance" — report "inner product on L2-normalized vectors"
# or "cosine similarity (via IndexFlatIP)".
FAISS_INDEX_TYPE = "IndexFlatIP"
FAISS_NORMALIZATION = "L2"  # applied to embeddings and query

# -----------------------------------------------------------------------------
# 5. ROUTING
# -----------------------------------------------------------------------------
# Two different metrics (must not be conflated):
#
# ROUTING_CLASSIFIER_ACCURACY = 1.0  — Classifier accuracy on train/test
#   (correct assignment to SLM vs LLM vs manual labels).
#
# ROUTING_END_TO_END_ACCURACY = 0.5667  — End-to-end answer quality (e.g. F1)
#   when using the routing strategy; not the same as classifier accuracy.
ROUTING_CLASSIFIER_ACCURACY = 1.0
ROUTING_END_TO_END_QUALITY = 0.5667  # e.g. F1 over routed answers

# -----------------------------------------------------------------------------
# 6. CO₂ (derived from NVML energy and grid intensity)
# -----------------------------------------------------------------------------
# Carbon intensity (U.S. average): kg CO₂e per kWh
CARBON_INTENSITY_KG_PER_KWH = 0.385

# CO₂ per query (kg CO₂e/query)
CO2_KG_PER_QUERY_SLM_NORAG = ENERGY_KWH_PER_QUERY_SLM_NORAG * CARBON_INTENSITY_KG_PER_KWH
CO2_KG_PER_QUERY_SLM_RAG   = ENERGY_KWH_PER_QUERY_SLM_RAG * CARBON_INTENSITY_KG_PER_KWH
CO2_KG_PER_QUERY_LLM_NORAG = ENERGY_KWH_PER_QUERY_LLM_NORAG * CARBON_INTENSITY_KG_PER_KWH
CO2_KG_PER_QUERY_LLM_RAG   = ENERGY_KWH_PER_QUERY_LLM_RAG * CARBON_INTENSITY_KG_PER_KWH
CO2_KG_PER_QUERY_ROUTING   = ENERGY_KWH_PER_QUERY_ROUTING * CARBON_INTENSITY_KG_PER_KWH

# CO₂ per million queries (kg CO₂e / 1M queries) — matches Results section.
CO2_KG_PER_MILLION_SLM_NORAG = 19.83   # approx
CO2_KG_PER_MILLION_SLM_RAG   = 19.82
CO2_KG_PER_MILLION_LLM_NORAG = 42.35
CO2_KG_PER_MILLION_LLM_RAG   = 42.47
CO2_KG_PER_MILLION_ROUTING   = 38.19

# Example annual workload for scaling (used only when explicitly stated).
CO2_ANNUAL_EXAMPLE_QUERIES_PER_YEAR = 117e6
