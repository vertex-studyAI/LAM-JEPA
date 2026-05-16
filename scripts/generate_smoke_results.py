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

import json
from lam_jepa.benchmarking.runner import train_variant, evaluate_tasks


def main():
    model, cfg, trainer = train_variant("full", steps=20, batch_size=16, seed=7, device="cpu")
    results = evaluate_tasks(model, cfg, batches=2, batch_size=16)
    payload = {"tasks": results, "history": trainer.history[-3:]}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
