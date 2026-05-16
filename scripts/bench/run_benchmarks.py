from __future__ import annotations

import argparse
import json
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

from lam_jepa.benchmarking.runner import train_variant, evaluate_tasks, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite.")
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="outputs/benchmark_results.json")
    args = parser.parse_args()
    model, cfg, trainer = train_variant(args.variant, steps=args.steps, batch_size=args.batch_size, seed=args.seed, device=args.device)
    task_scores = evaluate_tasks(model, cfg, batch_size=args.batch_size, batches=8)
    payload = {"variant": args.variant, "seed": args.seed, "steps": args.steps, "tasks": task_scores, "history_tail": trainer.history[-5:]}
    save_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
