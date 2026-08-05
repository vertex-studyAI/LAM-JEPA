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
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.baselines import evaluate_label_baselines
from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure deterministic label-distribution reference baselines for every benchmark task."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=LAMJEPAConfig().vocab_size)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval_baselines.json"))
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.batches < 1:
        parser.error("--batches must be at least 1")
    if args.vocab_size < 1:
        parser.error("--vocab-size must be at least 1")

    set_seed(args.seed)
    scores = evaluate_label_baselines(
        tasks=EDTECH_TASKS,
        batch_size=args.batch_size,
        batches=args.batches,
        vocab_size=args.vocab_size,
    )
    payload = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "vocab_size": args.vocab_size,
        "tasks": list(EDTECH_TASKS),
        "scores": scores,
        "interpretation": {
            "majority_accuracy": "oracle frequency of the most common label in the sampled evaluation rows",
            "uniform_observed_label_accuracy": "expected accuracy when guessing uniformly over labels observed in the sampled rows",
            "uniform_full_vocab_accuracy": "expected accuracy when guessing uniformly over the configured output vocabulary",
            "warning": "these are sampler references, not evidence of educational effectiveness or model quality",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
