from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_rows(rows: object, expected_ids: list[str] | None, name: str):
    require(isinstance(rows, list) and rows, f"{name}: prediction rows missing")
    ids: list[str] = []
    counts: list[int] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        require(isinstance(row, dict), f"{name}: row must be an object")
        item_id = str(row.get("id", ""))
        count = row.get("choice_count")
        label = row.get("label")
        prediction = row.get("prediction")
        values = row.get("probabilities")
        require(item_id, f"{name}: missing id")
        require(isinstance(count, int) and count >= 2, f"{name}/{item_id}: invalid choice count")
        require(isinstance(label, int) and 0 <= label < count, f"{name}/{item_id}: invalid label")
        require(isinstance(prediction, int) and 0 <= prediction < count, f"{name}/{item_id}: invalid prediction")
        require(isinstance(values, list) and len(values) == count, f"{name}/{item_id}: probability length != choice count")
        values = [float(value) for value in values]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), f"{name}/{item_id}: invalid probability")
        require(math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{name}/{item_id}: probabilities do not sum to one")
        require(prediction == max(range(count), key=values.__getitem__), f"{name}/{item_id}: prediction is not argmax")
        ids.append(item_id)
        counts.append(count)
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(values)
    require(len(ids) == len(set(ids)), f"{name}: duplicate ids")
    if expected_ids is not None:
        require(ids == expected_ids, f"{name}: row identity/order mismatch")

    n = len(ids)
    correct = [float(prediction == label) for prediction, label in zip(predictions, labels, strict=True)]
    brier = statistics.fmean(
        sum((value - (1.0 if index == label else 0.0)) ** 2 for index, value in enumerate(values))
        for values, label in zip(probabilities, labels, strict=True)
    )
    true_probability = statistics.fmean(values[label] for values, label in zip(probabilities, labels, strict=True))
    confidence = [max(values) for values in probabilities]
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
        "accuracy": float(statistics.fmean(correct)),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(true_probability),
    }, ids, counts, labels


def verify_metrics(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: metrics missing")
    for key, expected_value in expected.items():
        require(key in actual, f"{name}: missing metric {key}")
        require(
            math.isclose(float(actual[key]), expected_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{name}: {key} mismatch",
        )


def check_reversal(original: list[dict], reversed_rows: list[dict], name: str) -> float:
    require([row["id"] for row in original] == [row["id"] for row in reversed_rows], f"{name}: reversal changed row order")
    maximum = 0.0
    for left, right in zip(original, reversed_rows, strict=True):
        count = int(left["choice_count"])
        require(int(right["choice_count"]) == count, f"{name}/{left['id']}: reversal changed cardinality")
        require(int(right["label"]) == count - 1 - int(left["label"]), f"{name}/{left['id']}: reversal label mismatch")
        expected = list(reversed([float(value) for value in left["probabilities"]]))
        actual = [float(value) for value in right["probabilities"]]
        for actual_value, expected_value in zip(actual, expected, strict=True):
            maximum = max(maximum, abs(actual_value - expected_value))
    return maximum


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(actual: object, expected: dict, name: str) -> None:
    require(isinstance(actual, dict), f"{name}: summary missing")
    require(actual.get("n") == expected["n"], f"{name}: n mismatch")
    for key in ("mean", "std"):
        require(math.isclose(float(actual[key]), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9), f"{name}: {key} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify protocol-v3 variable-choice ARC smoke evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.results)
    protocol_file = read_json(args.protocol)
    protocol = payload.get("protocol")
    records = payload.get("records")
    majority = payload.get("majority_reference")
    summary = payload.get("summary")
    require(payload.get("artifact_type") == "lam-jepa ARC protocol-v3 variable-choice development smoke", "artifact type changed")
    require(isinstance(protocol, dict) and isinstance(records, list) and isinstance(majority, dict) and isinstance(summary, dict), "artifact structure invalid")

    require(protocol_file.get("protocol_id") == "lam-jepa-arc-challenge-v3", "protocol file id changed")
    require(protocol_file.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol not frozen")
    require(protocol.get("protocol_id") == protocol_file["protocol_id"], "artifact protocol id mismatch")
    require(protocol.get("protocol_sha256") == sha256_file(args.protocol), "artifact protocol hash mismatch")
    require(protocol.get("protocol_status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "artifact protocol status mismatch")
    require(protocol.get("answer_interface") == "variable-choice shared candidate scorer", "variable-choice answer interface missing")
    require(protocol.get("development_smoke_only") is True, "development smoke boundary missing")
    require(protocol.get("confirmatory_test_accessed") is False, "confirmatory test was accessed")
    require(protocol.get("all_selected_rows_retained") is True, "row-retention evidence missing")
    require(protocol.get("candidate_position_feature_used") is False, "candidate position feature must remain disabled")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")

    train_dist = protocol.get("train_choice_count_distribution")
    validation_dist = protocol.get("validation_choice_count_distribution")
    require(isinstance(train_dist, dict) and isinstance(validation_dist, dict), "choice distributions missing")
    require(any(key != "4" and value > 0 for key, value in validation_dist.items()), "smoke did not exercise non-four-choice validation rows")
    require(int(protocol.get("validation_examples", 0)) == sum(int(value) for value in validation_dist.values()), "validation distribution does not cover every selected row")
    require(int(protocol.get("train_examples", 0)) == sum(int(value) for value in train_dist.values()), "train distribution does not cover every selected row")

    majority_metrics, majority_ids, majority_counts, _ = metrics_from_rows(majority.get("predictions"), None, "majority")
    verify_metrics(majority.get("metrics"), majority_metrics, "majority")
    _, reverse_majority_ids, reverse_majority_counts, _ = metrics_from_rows(majority.get("choice_reversal_predictions"), majority_ids, "majority-reversed")
    require(reverse_majority_ids == majority_ids and reverse_majority_counts == majority_counts, "majority reversal changed row/cardinality identity")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid seed set")
    require(len(records) == len(seeds), "record count mismatch")
    canonical_ids: list[str] | None = None
    canonical_counts: list[int] | None = None
    canonical_labels: list[int] | None = None
    lam_accuracy: list[float] = []
    hash_accuracy: list[float] = []
    lam_max_error = 0.0
    hash_max_error = 0.0

    for expected_seed, record in zip(seeds, records, strict=True):
        require(isinstance(record, dict) and record.get("seed") == expected_seed, "seed record mismatch")
        lam = record.get("lam_jepa")
        hashed = record.get("hash_supervised")
        require(isinstance(lam, dict) and isinstance(hashed, dict), "model record missing")

        lam_metrics, ids, counts, labels = metrics_from_rows(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        if canonical_ids is None:
            canonical_ids = ids
            canonical_counts = counts
            canonical_labels = labels
        else:
            require(counts == canonical_counts and labels == canonical_labels, f"seed {expected_seed}: LAM row semantics changed")
        hash_metrics, hash_ids, hash_counts, hash_labels = metrics_from_rows(hashed.get("predictions"), canonical_ids, f"seed {expected_seed}/hash")
        require(hash_ids == canonical_ids and hash_counts == canonical_counts and hash_labels == canonical_labels, f"seed {expected_seed}: cross-model row semantics differ")
        require(len(ids) == int(protocol["validation_examples"]), f"seed {expected_seed}: missing validation rows")

        verify_metrics(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metrics(hashed.get("metrics"), hash_metrics, f"seed {expected_seed}/hash")
        lam_reverse_metrics, _, _, _ = metrics_from_rows(lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed")
        hash_reverse_metrics, _, _, _ = metrics_from_rows(hashed.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/hash-reversed")
        verify_metrics(lam.get("choice_reversal_metrics"), lam_reverse_metrics, f"seed {expected_seed}/lam-reversed")
        verify_metrics(hashed.get("choice_reversal_metrics"), hash_reverse_metrics, f"seed {expected_seed}/hash-reversed")

        lam_error = check_reversal(lam["predictions"], lam["choice_reversal_predictions"], f"seed {expected_seed}/lam")
        hash_error = check_reversal(hashed["predictions"], hashed["choice_reversal_predictions"], f"seed {expected_seed}/hash")
        require(math.isclose(float(lam.get("reversal_probability_max_abs_error")), lam_error, rel_tol=1e-9, abs_tol=1e-9), "LAM reversal error record mismatch")
        require(math.isclose(float(hashed.get("reversal_probability_max_abs_error")), hash_error, rel_tol=1e-9, abs_tol=1e-9), "hash reversal error record mismatch")
        lam_max_error = max(lam_max_error, lam_error)
        hash_max_error = max(hash_max_error, hash_error)
        lam_accuracy.append(lam_metrics["accuracy"])
        hash_accuracy.append(hash_metrics["accuracy"])

    require(Counter(canonical_counts or []) == Counter({int(key): int(value) for key, value in validation_dist.items()}), "raw row choice counts do not match protocol distribution")
    verify_summary(summary.get("lam_accuracy"), mean_std(lam_accuracy), "lam_accuracy")
    verify_summary(summary.get("hash_supervised_accuracy"), mean_std(hash_accuracy), "hash_accuracy")
    require(math.isclose(float(summary.get("lam_reversal_probability_max_abs_error")), lam_max_error, rel_tol=1e-9, abs_tol=1e-9), "LAM reversal summary mismatch")
    require(math.isclose(float(summary.get("hash_reversal_probability_max_abs_error")), hash_max_error, rel_tol=1e-9, abs_tol=1e-9), "hash reversal summary mismatch")
    require(lam_max_error <= 1e-5, f"LAM candidate scorer is not reversal-equivariant: {lam_max_error}")
    require(hash_max_error <= 1e-5, f"hash candidate scorer is not reversal-equivariant: {hash_max_error}")

    report = {
        "verdict": "VARIABLE_CHOICE_ARC_EXECUTION_VERIFIED_ONLY",
        "protocol_id": protocol["protocol_id"],
        "training_seeds": seeds,
        "train_examples": protocol["train_examples"],
        "validation_examples": protocol["validation_examples"],
        "train_choice_count_distribution": train_dist,
        "validation_choice_count_distribution": validation_dist,
        "all_selected_rows_retained": True,
        "non_four_choice_validation_exercised": True,
        "candidate_position_feature_used": False,
        "lam_reversal_probability_max_abs_error": lam_max_error,
        "hash_reversal_probability_max_abs_error": hash_max_error,
        "confirmatory_test_accessed": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
