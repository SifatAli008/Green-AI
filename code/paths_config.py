"""
Path config for local vs Kaggle. Use DATA_DIR for inputs, OUTPUT_DIR for outputs.
On Kaggle: data in /kaggle/input/<dataset>, results in /kaggle/working.
Set KAGGLE_INPUT_DATASET to your dataset name (default: green-paper-eval).
"""
import os

KAGGLE = os.path.exists("/kaggle")
KAGGLE_INPUT_DATASET = os.environ.get("KAGGLE_INPUT_DATASET", "green-paper-eval")

if KAGGLE:
    DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", f"/kaggle/input/{KAGGLE_INPUT_DATASET}")
    OUTPUT_DIR = os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working")
else:
    DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", ".")
    OUTPUT_DIR = os.environ.get("KAGGLE_OUTPUT_DIR", ".")

def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)

def output_path(*parts):
    return os.path.join(OUTPUT_DIR, *parts)
