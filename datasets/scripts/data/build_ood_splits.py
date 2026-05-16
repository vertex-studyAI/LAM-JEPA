from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets" / "external" / "math" / "gsm8k"
OUT = ROOT / "datasets" / "ood" / "reasoning_depth"
OUT.mkdir(parents=True, exist_ok=True)

train, test = [], []
train_path = SRC / "train"
if train_path.exists():
    for item in train_path.rglob("*"):
        pass

# This script intentionally expects the downloaded GSM8K data to be in dataset-disk format
# or for the user to supply jsonl files from the official source.
print("OOD split builder is ready; connect it to the exact GSM8K source format you use.")
