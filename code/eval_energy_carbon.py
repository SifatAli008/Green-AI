"""
Energy and carbon accounting for evaluation.
- Uses measurement_config for single source of truth.
- Computes per-query energy (token-based proxy) and CO2e.
- Optional: try NVML for GPU power if available (reports fallback if not).
"""
import os
import json
from typing import Dict, List, Optional

try:
    from measurement_config import (
        ENERGY_KWH_PER_TOKEN,
        CARBON_INTENSITY_KG_PER_KWH,
        CO2_SAVINGS_KG_PER_MILLION_QUERIES,
    )
except ImportError:
    ENERGY_KWH_PER_TOKEN = 0.00001
    CARBON_INTENSITY_KG_PER_KWH = 0.385
    CO2_SAVINGS_KG_PER_MILLION_QUERIES = 7.05

def energy_kwh_from_tokens(n_tokens: int) -> float:
    """Per-query energy (kWh) from generated token count (proxy)."""
    return n_tokens * ENERGY_KWH_PER_TOKEN

def carbon_kg_from_energy(kwh: float) -> float:
    """CO2e (kg) from energy (kWh) using U.S. grid intensity."""
    return kwh * CARBON_INTENSITY_KG_PER_KWH

def carbon_savings_per_million_queries(baseline_kwh_per_query: float, our_kwh_per_query: float) -> float:
    """kg CO2e saved per million queries = (baseline - our) * 1e6 * intensity."""
    saved_kwh = (baseline_kwh_per_query - our_kwh_per_query) * 1e6
    return saved_kwh * CARBON_INTENSITY_KG_PER_KWH

def aggregate_energy_carbon(
    results: List[Dict],
    token_key: str = "response_tokens",
) -> Dict:
    """
    results: list of dicts with response_tokens (or token_key).
    Returns: total_tokens, total_kwh, total_kg_co2e, mean_kwh_per_query.
    """
    tokens = [r.get(token_key, 0) for r in results if isinstance(r.get(token_key), (int, float))]
    total_tok = sum(tokens)
    n = len(tokens)
    total_kwh = energy_kwh_from_tokens(total_tok)
    total_kg = carbon_kg_from_energy(total_kwh)
    mean_kwh = total_kwh / n if n else 0
    return {
        "total_tokens": total_tok,
        "n_queries": n,
        "total_kwh": round(total_kwh, 6),
        "total_kg_co2e": round(total_kg, 6),
        "mean_kwh_per_query": round(mean_kwh, 6),
        "carbon_intensity_kg_per_kwh": CARBON_INTENSITY_KG_PER_KWH,
        "note": "Energy from token-based proxy; not hardware power measurement.",
    }

def try_nvml_power() -> Optional[float]:
    """If pynvml available and GPU present, return current power draw (W). Else None."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
        pynvml.nvmlShutdown()
        return w
    except Exception:
        return None

if __name__ == "__main__":
    # Example: 100 queries, 80 tokens each
    fake_results = [{"response_tokens": 80} for _ in range(100)]
    agg = aggregate_energy_carbon(fake_results)
    print("Energy/carbon aggregate:", json.dumps(agg, indent=2))
    w = try_nvml_power()
    print("NVML power (W):", w if w is not None else "not available")
