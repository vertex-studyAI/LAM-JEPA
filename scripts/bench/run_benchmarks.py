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

from lam_jepa.analysis.statistics import summarize_seed_runs
from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.benchmarking.runner import evaluate_tasks, save_json, train_variant
from lam_jepa.utils import set_seed


def _parse_seeds(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must contain unique training seeds")
        return list(args.seeds)
    if args.seed is not None:
        return [args.seed]
    return [7]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LAM-JEPA benchmark suite.")
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=1007)
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
    if args.eval_batches <= 0:
        parser.error("--eval-batches must be positive")

    training_seeds = _parse_seeds(args, parser)
    records: list[dict[str, Any]] = []
    per_task: dict[str, list[float]] = {task: [] for task in EDTECH_TASKS}
    expected_digests: dict[str, str] | None = None

    for training_seed in training_seeds:
        model, cfg, trainer = train_variant(
            args.variant,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=training_seed,
            device=args.device,
        )

        # Reset the benchmark sampler before every evaluation so all trained
        # models see the same ordered rows. Training-seed variance must not be
        # silently mixed with evaluation-sample variance.
        set_seed(args.evaluation_seed)
        task_scores = evaluate_tasks(
            model,
            cfg,
            batch_size=args.batch_size,
            batches=args.eval_batches,
        )
        digests = {
            task: str(metrics["sample_digest"])
            for task, metrics in task_scores.items()
        }
        if expected_digests is None:
            expected_digests = digests
        elif digests != expected_digests:
            raise RuntimeError("evaluation sample digests changed across training seeds")

        records.append(
            {
                "training_seed": training_seed,
                "evaluation_seed": args.evaluation_seed,
                "tasks": task_scores,
                "history_tail": trainer.history[-5:],
            }
        )
        for task, metrics in task_scores.items():
            per_task[task].append(float(metrics["accuracy"]))

    payload: dict[str, Any] = {
        "protocol": {
            "variant": args.variant,
            "training_seeds": training_seeds,
            "evaluation_seed": args.evaluation_seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "device": args.device,
            "tasks": list(EDTECH_TASKS),
            "evaluation_pairing": "identical ordered evaluation rows across training seeds",
        },
        "sample_digests": expected_digests or {},
        "records": records,
        "aggregate": summarize_seed_runs(per_task),
        "claim_boundary": (
            "Benchmark-runner outputs are execution records. Scientific claims require a predeclared "
            "dataset/evaluation protocol, adequate training budget, fair baselines, multiple seeds, "
            "statistical analysis, ablations, robustness checks, and independent reproduction."
        ),
    }

    # Preserve the historical single-seed fields for downstream consumers.
    if len(records) == 1:
        payload["variant"] = args.variant
        payload["seed"] = records[0]["training_seed"]
        payload["steps"] = args.steps
        payload["tasks"] = records[0]["tasks"]
        payload["history_tail"] = records[0]["history_tail"]

    save_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
