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

from lam_jepa.benchmarking.edtech_suite import ablation_suite, save_json


def main():
    p = argparse.ArgumentParser(
        description="Run paired, multi-seed, mechanism-exercising component ablations."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument(
        "--model-steps",
        type=int,
        default=1,
        help="Latent-action rollout steps. Must be >=1 so the planner ablation is identifiable.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=6)
    p.add_argument("--evaluation-seed", type=int, default=1007)
    p.add_argument("--training-task", type=str, default="mixed")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="outputs/ablation_results.json")
    args = p.parse_args()

    results = ablation_suite(
        seeds=args.seeds,
        steps=args.steps,
        model_steps=args.model_steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        evaluation_seed=args.evaluation_seed,
        task=args.training_task,
        device=args.device,
    )
    save_json(args.out, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
