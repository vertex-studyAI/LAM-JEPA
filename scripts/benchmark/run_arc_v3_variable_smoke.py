from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from lam_jepa.benchmarking.arc_challenge import load_arc_split, reverse_choices
from lam_jepa.benchmarking.arc_variable_choice import (
    choice_count_distribution,
    predict_cardinality_majority,
    predict_variable_hash,
    predict_variable_lam,
    train_variable_hash,
    train_variable_lam,
    variable_protocol_identity,
)
from lam_jepa.model import LAMJEPAConfig

EXPECTED_PROTOCOL_ID = "lam-jepa-arc-challenge-v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise SystemExit(f"expected protocol {EXPECTED_PROTOCOL_ID}")
    if protocol.get("status") != "FROZEN_BEFORE_CONFIRMATORY_TEST":
        raise SystemExit("protocol v3 is not frozen")
    interface = protocol.get("answer_interface", {})
    if interface.get("type") != "variable-choice shared candidate scorer":
        raise SystemExit("protocol v3 answer interface drifted")
    return protocol


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def reversal_error(original_rows: list[dict], reversed_rows: list[dict]) -> float:
    if [row["id"] for row in original_rows] != [row["id"] for row in reversed_rows]:
        raise RuntimeError("choice reversal changed item identity/order")
    maximum = 0.0
    for original, reversed_row in zip(original_rows, reversed_rows, strict=True):
        original_probabilities = [float(value) for value in original["probabilities"]]
        reversed_probabilities = [float(value) for value in reversed_row["probabilities"]]
        if len(original_probabilities) != len(reversed_probabilities):
            raise RuntimeError("choice reversal changed choice cardinality")
        expected_label = len(original_probabilities) - 1 - int(original["label"])
        if int(reversed_row["label"]) != expected_label:
            raise RuntimeError("choice reversal label remapping is incorrect")
        expected_probabilities = list(reversed(original_probabilities))
        for actual, expected in zip(reversed_probabilities, expected_probabilities, strict=True):
            maximum = max(maximum, abs(actual - expected))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protocol-v3 variable-choice ARC development smoke.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-non-four-validation", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    if args.learning_rate != float(protocol["training_budget"]["lam_jepa_learning_rate"]):
        parser.error("LAM learning rate must match frozen protocol v3")
    if args.model_steps != int(protocol["training_budget"]["model_steps"]):
        parser.error("model_steps must match frozen protocol v3")
    if min(args.epochs, args.batch_size, args.model_steps) < 1:
        parser.error("positive execution arguments required")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        parser.error("development smoke requires at least two unique seeds")

    train_all = load_arc_split(args.train)
    validation_all = load_arc_split(args.validation)
    train = list(train_all[: args.train_limit] if args.train_limit else train_all)
    validation = list(validation_all[: args.validation_limit] if args.validation_limit else validation_all)
    if not train or not validation:
        parser.error("train and validation must be non-empty")
    identity = variable_protocol_identity(train, validation)
    if args.require_non_four_validation and set(identity["validation_choice_count_distribution"]) == {"4"}:
        raise SystemExit("smoke failed to include a non-four-choice validation row")

    cfg = LAMJEPAConfig()
    reversed_validation = [reverse_choices(example) for example in validation]
    majority_metrics, majority_rows, majority_by_count, unseen_majority_counts = predict_cardinality_majority(train, validation)
    majority_reverse_metrics, majority_reverse_rows, _, _ = predict_cardinality_majority(train, reversed_validation)

    records: list[dict] = []
    lam_accuracies: list[float] = []
    hash_accuracies: list[float] = []
    lam_reversal_errors: list[float] = []
    hash_reversal_errors: list[float] = []

    for seed in seeds:
        lam = train_variable_lam(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_steps=args.model_steps,
            device=args.device,
        )
        baseline = train_variable_hash(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=float(protocol["training_budget"]["matched_baseline_learning_rate"]),
            device=args.device,
        )
        lam_metrics, lam_rows = predict_variable_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        lam_reverse_metrics, lam_reverse_rows = predict_variable_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        hash_metrics, hash_rows = predict_variable_hash(
            baseline,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            device=args.device,
        )
        hash_reverse_metrics, hash_reverse_rows = predict_variable_hash(
            baseline,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            device=args.device,
        )
        lam_error = reversal_error(lam_rows, lam_reverse_rows)
        hash_error = reversal_error(hash_rows, hash_reverse_rows)
        if not math.isfinite(lam_error) or not math.isfinite(hash_error):
            raise RuntimeError("non-finite reversal error")
        lam_reversal_errors.append(lam_error)
        hash_reversal_errors.append(hash_error)
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        hash_accuracies.append(float(hash_metrics["accuracy"]))
        records.append(
            {
                "seed": seed,
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_reverse_metrics,
                    "reversal_probability_max_abs_error": lam_error,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_reverse_rows,
                },
                "hash_supervised": {
                    "metrics": hash_metrics,
                    "choice_reversal_metrics": hash_reverse_metrics,
                    "reversal_probability_max_abs_error": hash_error,
                    "predictions": hash_rows,
                    "choice_reversal_predictions": hash_reverse_rows,
                },
            }
        )

    payload = {
        "artifact_type": "lam-jepa ARC protocol-v3 variable-choice development smoke",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(args.protocol),
            "protocol_status": protocol["status"],
            "answer_interface": protocol["answer_interface"]["type"],
            "development_smoke_only": True,
            "confirmatory_test_accessed": False,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size_questions": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_steps": args.model_steps,
            **identity,
            "all_selected_rows_retained": True,
            "candidate_position_feature_used": False,
            "majority_by_choice_count": {str(key): value for key, value in sorted(majority_by_count.items())},
            "majority_unseen_choice_counts": unseen_majority_counts,
            "claim_boundary": (
                "This development smoke verifies variable-choice execution on train/validation only. It does not execute the frozen five-seed/20-epoch budget, strong DeBERTa comparison, matched-capacity protocol-v3 baseline, required ablations, negative control, or confirmatory test."
            ),
        },
        "majority_reference": {
            "metrics": majority_metrics,
            "choice_reversal_metrics": majority_reverse_metrics,
            "predictions": majority_rows,
            "choice_reversal_predictions": majority_reverse_rows,
        },
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_accuracies),
            "hash_supervised_accuracy": summarize(hash_accuracies),
            "lam_reversal_probability_max_abs_error": max(lam_reversal_errors),
            "hash_reversal_probability_max_abs_error": max(hash_reversal_errors),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
