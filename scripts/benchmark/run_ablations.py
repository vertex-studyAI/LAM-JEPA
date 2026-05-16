from __future__ import annotations
import sys
from pathlib import Path

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import json

from lam_jepa.benchmarking.runner import ablation_suite, save_json


def main():
    p = argparse.ArgumentParser(description="Run ablation study.")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="outputs/ablation_results.json")
    args = p.parse_args()
    results = ablation_suite(steps=args.steps, batch_size=args.batch_size, device=args.device)
    save_json(args.out, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
