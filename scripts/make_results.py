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
    model, cfg, trainer = train_variant("full", steps=80, batch_size=32, seed=7, device="cpu")
    task_scores = evaluate_tasks(model, cfg, batches=4, batch_size=32)
    ablations = ablation_suite(steps=40, batch_size=32, device="cpu")
    payload = {
        "task_scores": task_scores,
        "ablations": ablations,
        "history_tail": trainer.history[-5:],
    }
    save_json(OUT / "summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
