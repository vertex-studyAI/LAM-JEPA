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
from pathlib import Path

from lam_jepa.benchmarking.runner import ablation_suite, evaluate_tasks, save_json, train_variant


OUT = Path("results")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, cfg, trainer = train_variant("full", steps=100, batch_size=64, seed=7, device="cpu")
    tasks = evaluate_tasks(model, cfg, batches=8, batch_size=64)
    ablations = ablation_suite(steps=60, batch_size=64, device="cpu")
    payload = {
        "tasks": tasks,
        "ablations": ablations,
        "history": trainer.history,
    }
    save_json(OUT / "extended_results.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
