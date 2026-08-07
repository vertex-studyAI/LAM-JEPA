from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"results not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "results must be a JSON object")
    return payload


def recompute_metrics(rows: object, expected_ids: list[str] | None, name: str) -> tuple[dict[str, float], list[str]]:
    require(isinstance(rows, list) and rows, f"{name}: predictions must be a non-empty list")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probability_rows: list[list[float]] = []

    for row in rows:
        require(isinstance(row, dict), f"{name}: prediction row must be an object")
        item_id = str(row.get("id", ""))
        require(item_id, f"{name}: missing item id")
        ids.append(item_id)

        label = row.get("label")
        prediction = row.get("prediction")
        require(isinstance(label, int) and 0 <= label < 4, f"{name}/{item_id}: invalid label")
        require(isinstance(prediction, int) and 0 <= prediction < 4, f"{name}/{item_id}: invalid prediction")
        labels.append(label)
        predictions.append(prediction)

        probabilities = row.get("probabilities")
        require(isinstance(probabilities, list) and len(probabilities) == 4, f"{name}/{item_id}: expected four probabilities")
        values = [float(value) for value in probabilities]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), f"{name}/{item_id}: invalid probability")
        require(math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{name}/{item_id}: probabilities do not sum to one")
        require(prediction == max(range(4), key=values.__getitem__), f"{name}/{item_id}: prediction does not match argmax")
        probability_rows.append(values)

    require(len(set(ids)) == len(ids), f"{name}: duplicate ids")
    if expected_ids is not None:
        require(ids == expected_ids, f"{name}: row identity/order mismatch")

    n = len(ids)
    accuracy = sum(int(prediction == label) for prediction, label in zip(predictions, labels, strict=True)) / n
    brier = 0.0
    true_probability = 0.0
    confidences: list[float] = []
    correct: list[float] = []
    for values, label, prediction in zip(probability_rows, labels, predictions, strict=True):
        for index, value in enumerate(values):
            target = 1.0 if index == label else 0.0
            brier += (value - target) ** 2
        true_probability += values[label]
        confidences.append(max(values))
        correct.append(float(prediction == label))
    brier /= n
    true_probability /= n

    ece = 0.0
    bins = 10
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if (confidence >= lower if index == 0 else confidence > lower) and confidence <= upper
        ]
        if members:
            mean_confidence = statistics.fmean(confidences[position] for position in members)
            mean_correct = statistics.fmean(correct[position] for position in members)
            ece += (len(members) / n) * abs(mean_correct - mean_confidence)

    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(true_probability),
    }, ids


def verify_metric_row(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: metrics missing")
    for key, expected_value in expected.items():
        require(key in actual, f"{name}: missing metric {key}")
        actual_value = float(actual[key])
        require(math.isfinite(actual_value), f"{name}: non-finite metric {key}")
        require(
            math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{name}: metric {key} mismatch: actual={actual_value} expected={expected_value}",
        )


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(actual: object, expected: dict[str, float | int], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: summary missing")
    require(actual.get("n") == expected["n"], f"{name}: seed count mismatch")
    for key in ("mean", "std"):
        require(
            math.isclose(float(actual[key]), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9),
            f"{name}: {key} mismatch",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify the ARC parameter-matched baseline artifact.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.results)
    protocol = payload.get("protocol")
    records = payload.get("records")
    summary = payload.get("summary")
    require(isinstance(protocol, dict), "protocol missing")
    require(isinstance(records, list), "records missing")
    require(isinstance(summary, dict), "summary missing")

    require(protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected dataset")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "test split boundary missing")
    for key in ("train_digest", "validation_digest", "train_id_digest", "validation_id_digest"):
        digest = protocol.get(key)
        require(isinstance(digest, str) and len(digest) == 64, f"invalid {key}")
    require(protocol["train_digest"] != protocol["validation_digest"], "train/validation dataset digests collide")
    require(protocol["train_id_digest"] != protocol["validation_id_digest"], "train/validation ID digests collide")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2, "verification smoke requires at least two seeds")
    require(len(set(seeds)) == len(seeds), "seeds must be unique")
    require(len(records) == len(seeds), "record count does not match seed count")
    require(protocol.get("primary_metric") == "multiple-choice accuracy", "primary metric changed")
    require(protocol.get("robustness_check") == "deterministic reversal of answer-choice order with label remapping", "robustness contract changed")

    lam_total = int(protocol.get("lam_total_trainable_parameters", 0))
    lam_active = int(protocol.get("lam_gradient_active_parameters", 0))
    matched_total = int(protocol.get("matched_supervised_trainable_parameters", 0))
    matched_active = int(protocol.get("matched_supervised_gradient_active_parameters", 0))
    require(lam_total > 0 and lam_active > 0 and matched_total > 0, "parameter counts must be positive")
    require(lam_active <= lam_total, "LAM active parameter count exceeds trainable total")
    require(matched_active == matched_total, "matched baseline contains inactive trainable padding")

    tolerance = float(protocol.get("parameter_match_tolerance", -1.0))
    declared_gap = float(protocol.get("parameter_relative_gap", -1.0))
    require(0.0 < tolerance < 1.0, "invalid parameter match tolerance")
    recomputed_gap = abs(matched_total - lam_active) / lam_active
    require(math.isclose(declared_gap, recomputed_gap, rel_tol=1e-12, abs_tol=1e-12), "parameter relative gap mismatch")
    require(recomputed_gap <= tolerance, f"parameter match exceeds tolerance: gap={recomputed_gap} tolerance={tolerance}")
    match_basis = str(protocol.get("parameter_match_basis", ""))
    require("gradient-active" in match_basis, "parameter match basis must remain gradient-active")
    require(protocol.get("strong_pretrained_baseline") == "NOT_INCLUDED", "strong pretrained baseline status must remain explicit")
    require(int(protocol.get("final_seed_requirement", 0)) >= 5, "final seed requirement weakened")

    claim_boundary = str(protocol.get("claim_boundary", ""))
    for phrase in (
        "strong pretrained",
        "locked-test",
        "independent reproduction",
        "RESEARCH_COMPLETE",
    ):
        require(phrase in claim_boundary, f"claim boundary missing required phrase: {phrase}")

    validation_n = int(protocol.get("validation_examples", 0))
    require(validation_n > 0, "validation set must be non-empty")
    canonical_ids: list[str] | None = None
    canonical_reversed_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    canonical_reversed_labels: list[int] | None = None
    lam_accuracies: list[float] = []
    matched_accuracies: list[float] = []
    deltas: list[float] = []

    for expected_seed, record in zip(seeds, records, strict=True):
        require(isinstance(record, dict), "seed record must be an object")
        require(record.get("seed") == expected_seed, "seed record order mismatch")

        lam = record.get("lam_jepa")
        matched = record.get("matched_supervised")
        require(isinstance(lam, dict), f"seed {expected_seed}: LAM record missing")
        require(isinstance(matched, dict), f"seed {expected_seed}: matched baseline record missing")

        lam_metrics, ids = recompute_metrics(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        if canonical_ids is None:
            canonical_ids = ids
        matched_metrics, matched_ids = recompute_metrics(matched.get("predictions"), canonical_ids, f"seed {expected_seed}/matched")
        require(matched_ids == canonical_ids, f"seed {expected_seed}: cross-model row mismatch")
        require(len(ids) == validation_n, f"seed {expected_seed}: validation row count mismatch")

        lam_reverse_metrics, reverse_ids = recompute_metrics(
            lam.get("choice_reversal_predictions"), canonical_reversed_ids, f"seed {expected_seed}/lam-reversed"
        )
        if canonical_reversed_ids is None:
            canonical_reversed_ids = reverse_ids
        matched_reverse_metrics, matched_reverse_ids = recompute_metrics(
            matched.get("choice_reversal_predictions"), canonical_reversed_ids, f"seed {expected_seed}/matched-reversed"
        )
        require(matched_reverse_ids == canonical_reversed_ids, f"seed {expected_seed}: reversed cross-model row mismatch")
        require(canonical_reversed_ids == canonical_ids, f"seed {expected_seed}: choice reversal changed item identity/order")

        lam_rows = lam["predictions"]
        matched_rows = matched["predictions"]
        lam_reverse_rows = lam["choice_reversal_predictions"]
        matched_reverse_rows = matched["choice_reversal_predictions"]
        lam_labels = [int(row["label"]) for row in lam_rows]
        matched_labels = [int(row["label"]) for row in matched_rows]
        lam_reverse_labels = [int(row["label"]) for row in lam_reverse_rows]
        matched_reverse_labels = [int(row["label"]) for row in matched_reverse_rows]
        require(lam_labels == matched_labels, f"seed {expected_seed}: cross-model validation labels differ")
        require(lam_reverse_labels == matched_reverse_labels, f"seed {expected_seed}: reversed cross-model labels differ")
        require(lam_reverse_labels == [3 - label for label in lam_labels], f"seed {expected_seed}: reversed label remapping is incorrect")

        if canonical_labels is None:
            canonical_labels = lam_labels
            canonical_reversed_labels = lam_reverse_labels
        else:
            require(lam_labels == canonical_labels, f"seed {expected_seed}: validation labels changed across seeds")
            require(lam_reverse_labels == canonical_reversed_labels, f"seed {expected_seed}: reversed labels changed across seeds")

        verify_metric_row(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metric_row(matched.get("metrics"), matched_metrics, f"seed {expected_seed}/matched")
        verify_metric_row(lam.get("choice_reversal_metrics"), lam_reverse_metrics, f"seed {expected_seed}/lam-reversed")
        verify_metric_row(matched.get("choice_reversal_metrics"), matched_reverse_metrics, f"seed {expected_seed}/matched-reversed")

        delta = float(lam_metrics["accuracy"] - matched_metrics["accuracy"])
        require(
            math.isclose(float(record.get("accuracy_delta_lam_minus_matched")), delta, rel_tol=1e-9, abs_tol=1e-9),
            f"seed {expected_seed}: paired accuracy delta mismatch",
        )
        lam_accuracies.append(lam_metrics["accuracy"])
        matched_accuracies.append(matched_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(summary.get("lam_accuracy"), mean_std(lam_accuracies), "lam_accuracy")
    verify_summary(summary.get("matched_supervised_accuracy"), mean_std(matched_accuracies), "matched_supervised_accuracy")
    verify_summary(summary.get("paired_accuracy_delta_lam_minus_matched"), mean_std(deltas), "paired_accuracy_delta")

    report = {
        "verdict": "CAPACITY_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "dataset": protocol["dataset"],
        "seeds": seeds,
        "validation_examples": validation_n,
        "lam_gradient_active_parameters": lam_active,
        "matched_supervised_gradient_active_parameters": matched_active,
        "parameter_relative_gap": recomputed_gap,
        "claim_boundary_preserved": True,
        "strong_pretrained_baseline_included": False,
        "locked_test_evaluated": False,
        "independent_reproduction": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
