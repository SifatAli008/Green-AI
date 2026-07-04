"""
Step-by-step test runner for evaluation and implementation.
Run: python run_all_tests.py
Kaggle: kaggle_run.py writes minimal queries/references under /kaggle/working if the dataset is missing.
"""
import sys
import os
import json

# Run from code folder (skip on Kaggle so /kaggle/working is cwd)
if not os.path.exists("/kaggle"):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

def step(name: str, fn):
    print("\n" + "=" * 60)
    print(f" STEP: {name}")
    print("=" * 60)
    try:
        fn()
        print(f" [PASS] {name}")
        return True
    except Exception as e:
        print(f" [FAIL] {name}: {e}")
        return False

def test_measurement_config():
    import measurement_config

    assert measurement_config.ENERGY_KWH_PER_QUERY_SLM_NORAG > 0
    assert measurement_config.CO2_KG_PER_MILLION_SLM_NORAG > 0
    print("  ENERGY_KWH_PER_QUERY_SLM_NORAG:", measurement_config.ENERGY_KWH_PER_QUERY_SLM_NORAG)
    print("  CO2_KG_PER_MILLION_SLM_NORAG:", measurement_config.CO2_KG_PER_MILLION_SLM_NORAG)

def test_quality_metrics():
    from eval_quality_metrics import compute_f1, load_references
    p, r, f1 = (0.5, 0.5, 0.5)  # placeholder
    out = compute_f1(
        "Metformin is first line treatment for diabetes.",
        "Metformin is first-line therapy for type 2 diabetes.",
    )
    assert "f1" in out and 0 <= out["f1"] <= 1
    print("  F1 sample:", out["f1"], "precision:", out["precision"], "recall:", out["recall"])
    try:
        from paths_config import data_path
        refs_path = data_path("references.json")
    except ImportError:
        refs_path = "references.json"
    refs = load_references(refs_path)
    print("  References loaded:", len(refs))

def test_safety_verification():
    from eval_safety_verification import evaluate_claims_in_output, aggregate_safety_metrics
    c = evaluate_claims_in_output(
        "Metformin is first line. Insulin is not required.",
        "Metformin is first-line therapy.",
    )
    assert "supported" in c and "contradicted" in c and "unsupported" in c
    agg = aggregate_safety_metrics([c])
    assert "pct_safe" in agg
    print("  Claim counts:", c)
    print("  Aggregate pct_safe:", agg["pct_safe"])

def test_routing():
    from eval_routing import evaluate_routing, rule_based_label
    qs = ["What is diabetes treatment?", "Management of acute stroke and differential diagnosis."] * 10
    out = evaluate_routing(qs, test_frac=0.3, seed=42)
    assert "test_accuracy" in out and "n_test" in out
    print("  Test accuracy:", out["test_accuracy"], "n_test:", out["n_test"])
    print("  Confusion matrix:", out["confusion_matrix"])

def test_energy_carbon():
    from eval_energy_carbon import energy_kwh_from_tokens, carbon_kg_from_energy, aggregate_energy_carbon
    kwh = energy_kwh_from_tokens(128)
    assert kwh > 0
    kg = carbon_kg_from_energy(kwh)
    assert kg > 0
    print("  128 tokens -> kWh:", kwh, "-> kg CO2e:", kg)
    agg = aggregate_energy_carbon([{"response_tokens": 80}, {"response_tokens": 120}])
    assert agg["n_queries"] == 2
    print("  Aggregate:", agg["mean_kwh_per_query"], "kWh/query")

def test_full_eval():
    from eval_full_run import main
    main()
    try:
        from paths_config import output_path
        results_path = output_path("eval_results.json")
    except ImportError:
        results_path = "eval_results.json"
    assert os.path.exists(results_path), "eval_results.json not written"
    print("  eval_results.json written")

def test_queries_and_references_exist():
    try:
        from paths_config import data_path
        qpath = data_path("queries_expanded.json")
        rpath = data_path("references.json")
    except ImportError:
        qpath, rpath = "queries_expanded.json", "references.json"
    if not os.path.exists(qpath):
        print("  queries_expanded.json: absent (eval_full_run uses built-in fallback queries)")
        if os.path.exists(rpath):
            with open(rpath, "r", encoding="utf-8") as f:
                print("  references.json: %d entries" % len(json.load(f)))
        else:
            print("  references.json: absent (optional for F1)")
        return
    with open(qpath, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert "queries" in d and len(d["queries"]) >= 12
    print("  queries_expanded.json: %d queries" % len(d["queries"]))
    if os.path.exists(rpath):
        with open(rpath, "r", encoding="utf-8") as f:
            print("  references.json: %d entries" % len(json.load(f)))
    else:
        print("  references.json: absent (optional for F1)")

if __name__ == "__main__":
    import json
    steps = [
        ("Measurement config", test_measurement_config),
        ("Queries and references files", test_queries_and_references_exist),
        ("Quality metrics (F1)", test_quality_metrics),
        ("Safety verification", test_safety_verification),
        ("Routing (train/test)", test_routing),
        ("Energy/carbon", test_energy_carbon),
        ("Full evaluation run", test_full_eval),
    ]
    passed = 0
    for name, fn in steps:
        if step(name, fn):
            passed += 1
    print("\n" + "=" * 60)
    print(f" RESULT: {passed}/{len(steps)} steps passed")
    print("=" * 60)
    sys.exit(0 if passed == len(steps) else 1)
