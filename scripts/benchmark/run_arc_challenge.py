from __future__ import annotations

import argparse
import json
from pathlib import Path

from lam_jepa.benchmarking.arc_challenge import load_arc_split, run_arc_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the externally grounded AI2 ARC-Challenge benchmark contract.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train = load_arc_split(args.train)
    validation = load_arc_split(args.validation)
    payload = run_arc_smoke(
        train,
        validation,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        model_steps=args.model_steps,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
