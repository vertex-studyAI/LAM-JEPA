from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

EXPECTED_PROTOCOL_ID = "lam-jepa-arc-challenge-v2"
EXPECTED_PERMUTATION_SEED = 20260807
EXPECTED_THRESHOLD = 0.35


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected object JSON: {path}")
    return payload


def recompute_metrics(rows: object, expected_ids: list[str] | None, name: str):
    require(isinstance(rows, list) and rows, f"{name}: predictions missing")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        require(isinstance(row, dict), f"{name}: row must be object")
        item_id = str(row.get("id", ""))
        label = row.get("label")
        prediction = row.get("prediction")
        values = row.get("probabilities")
        require(item_id, f"{name}: missing id")
        require(isinstance(label, int) and 0 <= label < 4, f"{name}/{item_id}: invalid label")
        require(isinstance(prediction, int) and 0 <= prediction < 4, f"{name}/{item_id}: invalid prediction")
        require(isinstance(values, list) and len(values) == 4, f"{name}/{item_id}: invalid probabilities")
        values = [float(value) for value in values]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), f"{name}/{item_id}: bad probability")
        require(math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{name}/{item_id}: probabilities do not sum to one")
        require(prediction == max(range(4), key=values.__getitem__), f"{name}/{item_id}: prediction is not argmax")
        ids.append(item_id)
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(values)
    require(len(ids) == len(set(ids)), f"{name}: duplicate ids")
    if expected_ids is not None:
        require(ids == expected_ids, f"{name}: row identity/order mismatch")

    n = len(ids)
    accuracy = sum(int(prediction == label) for prediction, label in zip(predictions, labels, strict=True)) / n
    brier = sum(
        sum((value - (1.0 if index == label else 0.0)) ** 2 for index, value in enumerate(values))
        for values, label in zip(probabilities, labels, strict=True)
    ) / n
    mean_true = statistics.fmean(values[label] for values, label in zip(probabilities, labels, strict=True))
    confidence = [max(values) for values in probabilities]
    correct = [float(prediction == label) for prediction, label in zip(predictions, labels, strict=True)]
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            index for index, conf in enumerate(confidence)
            if (conf >= lower if bin_index == 0 else conf > lower) and conf <= upper
        ]
        if members:
            ece += (len(members) / n) * abs(
                statistics.fmean(correct[index] for index in members)
                - statistics.fmean(confidence[index] for index in members)
            )
    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(mean_true),
    }, ids


def verify_metrics(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: metrics missing")
    for key, expected_value in expected.items():
        require(
            math.isclose(float(actual[key]), expected_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{name}: {key} mismatch",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARC shuffled-label negative-control evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.results)
    frozen = read_json(args.protocol)
    protocol = payload.get("protocol")
    records = payload.get("records")
    summary = payload.get("summary")
    require(payload.get("artifact_type") == "lam-jepa ARC protocol-v2 shuffled-label negative-control development smoke", "artifact type changed")
    require(isinstance(protocol, dict) and isinstance(records, list) and isinstance(summary, dict), "artifact structure invalid")

    require(frozen.get("protocol_id") == EXPECTED_PROTOCOL_ID, "frozen protocol id changed")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "frozen protocol is not frozen")
    negative = frozen.get("negative_control", {})
    require(negative.get("type") == "deterministic training-label permutation", "negative-control type changed")
    require(negative.get("permutation_seed") == EXPECTED_PERMUTATION_SEED, "permutation seed changed")
    require("0.35" in str(negative.get("failure_rule", "")), "failure threshold changed")

    require(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "artifact protocol id mismatch")
    require(protocol.get("protocol_sha256") == sha256_file(args.protocol), "protocol hash mismatch")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "artifact protocol status mismatch")
    require(protocol.get("development_smoke_only") is True, "development-smoke boundary missing")
    require(protocol.get("confirmatory_test_accessed") is False, "confirmatory test was accessed")
    require(protocol.get("negative_control_type") == negative.get("type"), "negative-control type drift")
    require(protocol.get("permutation_seed") == EXPECTED_PERMUTATION_SEED, "artifact permutation seed drift")
    require(math.isclose(float(protocol.get("failure_threshold_accuracy")), EXPECTED_THRESHOLD), "artifact threshold drift")
    require(float(protocol.get("learning_rate")) == float(frozen["training_budget"]["lam_jepa_learning_rate"]), "LAM LR drift")
    require(int(protocol.get("model_steps")) == int(frozen["training_budget"]["model_steps"]), "model_steps drift")
    require(protocol.get("train_id_digest") == protocol.get("permuted_train_id_digest"), "label permutation changed training row identity/order")
    require(protocol.get("original_train_digest") != protocol.get("permuted_train_digest"), "label permutation did not change training dataset digest")

    permutation = protocol.get("permutation")
    require(isinstance(permutation, dict), "permutation evidence missing")
    require(permutation.get("permutation_seed") == EXPECTED_PERMUTATION_SEED, "permutation evidence seed mismatch")
    mapping = permutation.get("mapping")
    require(isinstance(mapping, list) and mapping, "permutation mapping missing")
    original_labels = [int(row["original_label"]) for row in mapping]
    observed_permuted = [int(row["permuted_label"]) for row in mapping]
    expected_permuted = list(original_labels)
    random.Random(EXPECTED_PERMUTATION_SEED).shuffle(expected_permuted)
    require(observed_permuted == expected_permuted, "stored label permutation does not match deterministic seed")
    changed = sum(int(left != right) for left, right in zip(original_labels, observed_permuted, strict=True))
    require(changed == int(permutation.get("changed_label_count", -1)) and changed > 0, "changed-label count invalid")
    require(Counter(original_labels) == Counter(observed_permuted), "permutation changed label multiset")
    require(permutation.get("original_label_counts") == permutation.get("permuted_label_counts"), "stored label histograms differ")

    mapping_digest = hashlib.sha256()
    for row in mapping:
        normalized = {
            "id": str(row["id"]),
            "original_label": int(row["original_label"]),
            "permuted_label": int(row["permuted_label"]),
        }
        mapping_digest.update(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        mapping_digest.update(b"\n")
    require(mapping_digest.hexdigest() == permutation.get("mapping_digest"), "permutation mapping digest mismatch")

    seeds = protocol.get("training_seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid training seeds")
    require(len(records) == len(seeds), "record count does not match training seeds")
    validation_n = int(protocol.get("validation_examples", 0))
    require(validation_n > 0, "validation row count missing")
    canonical_ids: list[str] | None = None
    accuracies: list[float] = []
    exceeded: list[int] = []

    for expected_seed, record in zip(seeds, records, strict=True):
        require(record.get("training_seed") == expected_seed, "training seed record mismatch")
        metrics, ids = recompute_metrics(record.get("predictions"), canonical_ids, f"seed {expected_seed}")
        if canonical_ids is None:
            canonical_ids = ids
        require(len(ids) == validation_n, f"seed {expected_seed}: validation row count mismatch")
        verify_metrics(record.get("metrics"), metrics, f"seed {expected_seed}")
        accuracy = metrics["accuracy"]
        actual_exceeded = accuracy > EXPECTED_THRESHOLD
        require(bool(record.get("threshold_exceeded")) == actual_exceeded, f"seed {expected_seed}: threshold flag mismatch")
        if actual_exceeded:
            exceeded.append(expected_seed)
        accuracies.append(accuracy)

    expected_mean = statistics.fmean(accuracies)
    expected_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    aggregate = summary.get("validation_accuracy")
    require(isinstance(aggregate, dict), "validation-accuracy summary missing")
    require(aggregate.get("n") == len(accuracies), "summary seed count mismatch")
    require(math.isclose(float(aggregate["mean"]), expected_mean, rel_tol=1e-9, abs_tol=1e-9), "summary mean mismatch")
    require(math.isclose(float(aggregate["std"]), expected_std, rel_tol=1e-9, abs_tol=1e-9), "summary std mismatch")
    require(summary.get("threshold_exceeded_seeds") == exceeded, "threshold-exceeded seed list mismatch")
    require(bool(summary.get("failure_condition_triggered")) == bool(exceeded), "failure-condition summary mismatch")
    require(not exceeded, f"negative-control stop condition triggered: {exceeded}")

    report = {
        "verdict": "ARC_NEGATIVE_CONTROL_EXECUTION_VERIFIED_ONLY",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "permutation_seed": EXPECTED_PERMUTATION_SEED,
        "changed_label_count": changed,
        "training_seeds": seeds,
        "mean_validation_accuracy": expected_mean,
        "failure_threshold": EXPECTED_THRESHOLD,
        "failure_condition_triggered": False,
        "confirmatory_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
