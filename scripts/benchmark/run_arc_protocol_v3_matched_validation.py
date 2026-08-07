from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split, reverse_choices, score_predictions
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig
from scripts.benchmark.run_arc_matched_baseline import (
    choose_matched_architecture,
    predict_lam,
    predict_matched,
    probe_lam_active_parameters,
    train_matched_baseline,
)
from lam_jepa.benchmarking.arc_challenge import _train_lam_jepa

BOOTSTRAP_SEED = 20260807
BOOTSTRAP_DRAWS = 20000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    require(path.is_file(), f"protocol missing: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v3", "unexpected protocol id")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v3 is not frozen")
    require(protocol.get("dataset", {}).get("eligibility_rule") == "len(choices) == 4", "unexpected v3 eligibility rule")
    matched = protocol.get("models", {}).get("matched_capacity_supervised_baseline", {})
    require(matched.get("allowed_parameter_ratio_min") == 0.99, "matched lower parameter ratio drifted")
    require(matched.get("allowed_parameter_ratio_max") == 1.01, "matched upper parameter ratio drifted")
    require("gradient-active" in str(matched.get("parameter_accounting", "")), "matched accounting is not gradient-active")
    return protocol


def eligibility_summary(result) -> dict[str, object]:
    return {
        "source_count": result.original_count,
        "eligible_count": result.eligible_count,
        "excluded_count": result.excluded_count,
        "choice_count_distribution": {str(k): v for k, v in sorted(result.choice_count_distribution.items())},
        "eligible_id_digest": result.eligible_id_digest,
        "excluded_id_digest": result.excluded_id_digest,
        "excluded": [
            {"id": example.item_id, "choice_count": len(example.choices)}
            for example in result.excluded
        ],
    }


def bootstrap_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    lower_index = max(0, int(math.floor(0.025 * (draws - 1))))
    upper_index = min(draws - 1, int(math.ceil(0.975 * (draws - 1))))
    return [float(means[lower_index]), float(means[upper_index])]


def summarize(values: Sequence[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "ci95": bootstrap_ci(values),
    }


def prediction_diagnostics(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    probabilities = [[float(value) for value in row["probabilities"]] for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    unique_probabilities_6dp = len({tuple(round(value, 6) for value in row) for row in probabilities})
    per_class_ranges = [
        max(row[index] for row in probabilities) - min(row[index] for row in probabilities)
        for index in range(4)
    ]
    return {
        "prediction_class_count": len(set(predictions)),
        "prediction_histogram": {str(index): predictions.count(index) for index in range(4)},
        "unique_probability_rows_6dp": unique_probabilities_6dp,
        "per_class_probability_ranges": per_class_ranges,
        "maximum_probability_range": max(per_class_ranges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen protocol-v3 matched-capacity ARC train/validation package.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    budget = protocol["training_budget"]
    seeds = [int(seed) for seed in budget["training_seeds"]]
    require(seeds == [1, 2, 3, 4, 5], "frozen five-seed set drifted")
    epochs = int(budget["epochs"])
    batch_size = int(budget["batch_size"])
    lam_lr = float(budget["lam_jepa_learning_rate"])
    matched_lr = float(budget["matched_baseline_learning_rate"])
    model_steps = int(budget["model_steps"])
    require(epochs == 20 and batch_size == 32 and model_steps == 1, "frozen budget drifted")
    require(lam_lr == matched_lr == 3e-4, "frozen LAM/matched learning rate drifted")

    source_train = load_arc_split(args.train)
    source_validation = load_arc_split(args.validation)
    train_partition = select_protocol_eligible_examples(source_train)
    validation_partition = select_protocol_eligible_examples(source_validation)
    train = list(train_partition.eligible)
    validation = list(validation_partition.eligible)
    require(train_partition.original_count == 1119 and train_partition.eligible_count == 1117, "unexpected train eligibility counts")
    require(validation_partition.original_count == 299 and validation_partition.eligible_count == 295, "unexpected validation eligibility counts")
    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    require(not overlap, f"eligible train/validation overlap: {overlap[:5]}")

    cfg = LAMJEPAConfig()
    lam_active, lam_total = probe_lam_active_parameters(
        train,
        cfg=cfg,
        batch_size=batch_size,
        model_steps=model_steps,
        device=args.device,
    )
    depth, hidden_dim, baseline_total, parameter_gap = choose_matched_architecture(
        cfg,
        target_active_parameters=lam_active,
        tolerance=0.01,
    )
    parameter_ratio = baseline_total / lam_active
    require(0.99 <= parameter_ratio <= 1.01, f"matched parameter ratio outside frozen v3 gate: {parameter_ratio}")

    reversed_validation = [reverse_choices(example) for example in validation]
    records: list[dict[str, object]] = []
    lam_accuracies: list[float] = []
    matched_accuracies: list[float] = []
    paired_deltas: list[float] = []
    lam_robustness_drops: list[float] = []
    matched_robustness_drops: list[float] = []

    for seed in seeds:
        matched_start = time.perf_counter()
        matched, baseline_active = train_matched_baseline(
            train,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=matched_lr,
            device=args.device,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        matched_train_seconds = float(time.perf_counter() - matched_start)
        require(baseline_active == baseline_total, "matched baseline active parameter count changed")

        lam_start = time.perf_counter()
        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lam_lr,
            model_steps=model_steps,
            device=args.device,
        )
        lam_train_seconds = float(time.perf_counter() - lam_start)

        lam_eval_start = time.perf_counter()
        lam_probs, lam_labels, lam_rows = predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=args.device,
        )
        lam_eval_seconds = float(time.perf_counter() - lam_eval_start)
        matched_eval_start = time.perf_counter()
        matched_probs, matched_labels, matched_rows = predict_matched(
            matched,
            validation,
            cfg=cfg,
            batch_size=batch_size,
            device=args.device,
        )
        matched_eval_seconds = float(time.perf_counter() - matched_eval_start)
        require(lam_labels.tolist() == matched_labels.tolist(), "LAM and matched labels differ")

        lam_rev_probs, lam_rev_labels, lam_rev_rows = predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=args.device,
        )
        matched_rev_probs, matched_rev_labels, matched_rev_rows = predict_matched(
            matched,
            reversed_validation,
            cfg=cfg,
            batch_size=batch_size,
            device=args.device,
        )
        require(lam_rev_labels.tolist() == matched_rev_labels.tolist(), "LAM and matched reversed labels differ")

        lam_metrics = score_predictions(lam_probs, lam_labels)
        matched_metrics = score_predictions(matched_probs, matched_labels)
        lam_rev_metrics = score_predictions(lam_rev_probs, lam_rev_labels)
        matched_rev_metrics = score_predictions(matched_rev_probs, matched_rev_labels)
        delta = float(lam_metrics["accuracy"] - matched_metrics["accuracy"])
        lam_drop = float(lam_metrics["accuracy"] - lam_rev_metrics["accuracy"])
        matched_drop = float(matched_metrics["accuracy"] - matched_rev_metrics["accuracy"])
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        matched_accuracies.append(float(matched_metrics["accuracy"]))
        paired_deltas.append(delta)
        lam_robustness_drops.append(lam_drop)
        matched_robustness_drops.append(matched_drop)

        records.append(
            {
                "seed": seed,
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_rev_metrics,
                    "robustness_accuracy_drop": lam_drop,
                    "training_wall_seconds": lam_train_seconds,
                    "validation_wall_seconds": lam_eval_seconds,
                    "diagnostics": prediction_diagnostics(lam_rows),
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_rev_rows,
                },
                "matched_supervised": {
                    "metrics": matched_metrics,
                    "choice_reversal_metrics": matched_rev_metrics,
                    "robustness_accuracy_drop": matched_drop,
                    "training_wall_seconds": matched_train_seconds,
                    "validation_wall_seconds": matched_eval_seconds,
                    "diagnostics": prediction_diagnostics(matched_rows),
                    "predictions": matched_rows,
                    "choice_reversal_predictions": matched_rev_rows,
                },
                "accuracy_delta_lam_minus_matched": delta,
            }
        )

    paired_summary = summarize(paired_deltas)
    effect_threshold = float(protocol["metrics"]["practical_effect_threshold_absolute"])
    matched_superiority_gate = (
        float(paired_summary["mean"]) >= effect_threshold
        and float(paired_summary["ci95"][0]) > 0.0
    )
    lam_robustness_summary = summarize(lam_robustness_drops)
    matched_robustness_summary = summarize(matched_robustness_drops)
    robustness_limit = float(protocol["robustness"]["maximum_allowed_lam_accuracy_drop"])
    robustness_gate = (
        float(lam_robustness_summary["mean"]) <= robustness_limit
        and float(lam_robustness_summary["mean"]) - float(matched_robustness_summary["mean"]) <= 0.02
    )

    payload = {
        "artifact_type": "lam-jepa ARC protocol-v3 full matched-capacity validation package",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(args.protocol),
            "status": protocol["status"],
            "confirmatory_test_accessed": False,
            "eligibility_rule": protocol["dataset"]["eligibility_rule"],
            "train_eligibility": eligibility_summary(train_partition),
            "validation_eligibility": eligibility_summary(validation_partition),
            "train_examples_used": len(train),
            "validation_examples_used": len(validation),
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "training_seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lam_jepa_learning_rate": lam_lr,
            "matched_baseline_learning_rate": matched_lr,
            "model_steps": model_steps,
            "lam_total_trainable_parameters": lam_total,
            "lam_gradient_active_parameters": lam_active,
            "matched_supervised_trainable_parameters": baseline_total,
            "matched_supervised_gradient_active_parameters": baseline_total,
            "matched_depth": depth,
            "matched_hidden_dim": hidden_dim,
            "parameter_ratio_matched_to_lam_active": parameter_ratio,
            "parameter_relative_gap": parameter_gap,
            "claim_boundary": (
                "This is the frozen five-seed/20-epoch train+validation comparison against the v3 matched-capacity supervised baseline. "
                "It does not evaluate ARC test, does not include the frozen strong DeBERTa comparison, and cannot by itself authorize a headline superiority claim or RESEARCH_COMPLETE."
            ),
        },
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_accuracies),
            "matched_supervised_accuracy": summarize(matched_accuracies),
            "paired_accuracy_delta_lam_minus_matched": paired_summary,
            "lam_choice_reversal_accuracy_drop": lam_robustness_summary,
            "matched_choice_reversal_accuracy_drop": matched_robustness_summary,
            "matched_superiority_gate": matched_superiority_gate,
            "robustness_gate_against_matched": robustness_gate,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
