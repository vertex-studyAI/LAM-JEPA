from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

from lam_jepa.benchmarking.runner import evaluate_tasks, save_json, train_variant


def _parse_seeds(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must contain unique training seeds")
        return list(args.seeds)
    if args.seed is not None:
        return [args.seed]
    return [7]


def _run_one(
    *,
    variant: str,
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    model, cfg, trainer = train_variant(
        variant,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    task_scores = evaluate_tasks(model, cfg, batch_size=batch_size, batches=8)
    return {
        "seed": seed,
        "tasks": task_scores,
        "history_tail": trainer.history[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LAM-JEPA benchmark suite.")
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one training seed. Retained for backward compatibility.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple unique training seeds and retain one result record per seed.",
    )
    parser.add_argument("--out", type=str, default="outputs/benchmark_results.json")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    seeds = _parse_seeds(args, parser)
    runs = [
        _run_one(
            variant=args.variant,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        for seed in seeds
    ]

    payload: dict[str, Any] = {
        "variant": args.variant,
        "seeds": seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "runs": runs,
        "claim_boundary": (
            "Benchmark-runner outputs are execution records. Scientific claims require a predeclared "
            "dataset/evaluation protocol, adequate training budget, fair baselines, multiple seeds, "
            "statistical analysis, ablations, robustness checks, and independent reproduction."
        ),
    }

    # Preserve the historical single-seed top-level fields for downstream consumers while
    # making the multi-seed structure explicit through `runs` and `seeds`.
    if len(runs) == 1:
        payload["seed"] = runs[0]["seed"]
        payload["tasks"] = runs[0]["tasks"]
        payload["history_tail"] = runs[0]["history_tail"]

    save_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
