"""
Kaggle entrypoint: run full evaluation. Use from a Kaggle notebook or as main script.
- Reads data from /kaggle/input/<dataset> if present, else writes minimal data to /kaggle/working and runs.
- Set HF_TOKEN in Kaggle Secrets for real models; optional EVAL_USE_REAL_MODEL=1.
"""
import os
import sys
import json

# Ensure Kaggle paths when under /kaggle
if os.path.exists("/kaggle"):
    os.environ.setdefault("KAGGLE_OUTPUT_DIR", "/kaggle/working")
    dataset = os.environ.get("KAGGLE_INPUT_DATASET", "green-paper-eval")
    data_dir = f"/kaggle/input/{dataset}"
    if not os.path.exists(data_dir) or not os.path.exists(os.path.join(data_dir, "queries_expanded.json")):
        # No dataset or missing file: write minimal data to working dir
        data_dir = "/kaggle/working"
        os.environ["KAGGLE_DATA_DIR"] = data_dir
        minimal_queries = {
            "queries": [
                "What is the treatment for diabetes and its associated management?",
                "How Can AI Assist in the Diagnosis of Disease?",
                "What Are the Current Guidelines for the Management of Hypertension?",
                "Current ACS guidelines post-STEMI revascularization",
                "Management of acute ischemic stroke within 4.5 hours",
            ] * 4,
            "description": "Minimal set for Kaggle (no dataset attached)",
        }
        with open(os.path.join(data_dir, "queries_expanded.json"), "w", encoding="utf-8") as f:
            json.dump(minimal_queries, f, indent=2)
        refs = {
            "What is the treatment for diabetes and its associated management?": "Metformin is first-line. Lifestyle modification recommended.",
            "How Can AI Assist in the Diagnosis of Disease?": "AI can assist in diagnosis through decision support and imaging analysis.",
            "What Are the Current Guidelines for the Management of Hypertension?": "Lifestyle modification and pharmacotherapy based on cardiovascular risk.",
        }
        with open(os.path.join(data_dir, "references.json"), "w", encoding="utf-8") as f:
            json.dump(refs, f, indent=2)
        print("Wrote minimal queries_expanded.json and references.json to /kaggle/working")
    else:
        os.environ.setdefault("KAGGLE_DATA_DIR", data_dir)
    # Prepend dataset dir so we can import the repo modules
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = data_dir
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

def main():
    from run_all_tests import step, test_measurement_config, test_quality_metrics, test_safety_verification
    from run_all_tests import test_routing, test_energy_carbon, test_full_eval, test_queries_and_references_exist
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
    return passed == len(steps)

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
