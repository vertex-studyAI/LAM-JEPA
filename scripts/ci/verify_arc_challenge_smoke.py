from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def metric_row(row: object, name: str) -> dict:
    require(isinstance(row, dict), f"{name}: metrics must be an object")
    for key in ("accuracy", "brier", "ece", "mean_true_class_probability"):
        require(key in row, f"{name}: missing metric {key}")
        value = float(row[key])
        require(math.isfinite(value), f"{name}: non-finite metric {key}")
    require(0.0 <= float(row["accuracy"]) <= 1.0, f"{name}: accuracy out of range")
    require(float(row["brier"]) >= 0.0, f"{name}: negative Brier score")
    require(0.0 <= float(row["ece"]) <= 1.0, f"{name}: ECE out of range")
    require(0.0 <= float(row["mean_true_class_probability"]) <= 1.0, f"{name}: true-class probability out of range")
    return row


def verify_prediction_rows(rows: object, expected_n: int, expected_ids: list[str] | None, name: str) -> list[str]:
    require(isinstance(rows, list) and len(rows) == expected_n, f"{name}: raw prediction count mismatch")
    ids: list[str] = []
    for row in rows:
        require(isinstance(row, dict), f"{name}: prediction row must be an object")
        item_id = str(row.get("id", ""))
        require(item_id, f"{name}: missing item id")
        ids.append(item_id)
        label = row.get("label")
        prediction = row.get("prediction")
        require(isinstance(label, int) and 0 <= label < 4, f"{name}/{item_id}: invalid label")
        require(isinstance(prediction, int) and 0 <= prediction < 4, f"{name}/{item_id}: invalid prediction")
        probabilities = row.get("probabilities")
        require(isinstance(probabilities, list) and len(probabilities) == 4, f"{name}/{item_id}: probability vector must have four entries")
        values = [float(value) for value in probabilities]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), f"{name}/{item_id}: invalid probability")
        require(math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{name}/{item_id}: probabilities do not sum to one")
    require(len(set(ids)) == len(ids), f"{name}: duplicate validation ids")
    if expected_ids is not None:
        require(ids == expected_ids, f"{name}: validation row order changed")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify ARC-Challenge smoke evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.results.is_file(), f"results not found: {args.results}")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "results must be a JSON object")
    protocol = payload.get("protocol")
    require(isinstance(protocol, dict), "protocol missing")

    require(protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected dataset")
    require(protocol.get("test_split_policy") == "held out from this command", "test split policy missing or weakened")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    for key in ("train_digest", "validation_digest", "train_id_digest", "validation_id_digest"):
        digest = protocol.get(key)
        require(isinstance(digest, str) and len(digest) == 64, f"invalid {key}")
    require(protocol["train_digest"] != protocol["validation_digest"], "train and validation digests must differ")
    require(protocol["train_id_digest"] != protocol["validation_id_digest"], "train and validation ID digests must differ")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2, "CI smoke requires at least two unique seeds")
    require(len(set(seeds)) == len(seeds), "training seeds must be unique")
    require(isinstance(protocol.get("model_steps"), int) and protocol["model_steps"] >= 1, "LAM benchmark must exercise planner steps")
    validation_n = int(protocol.get("validation_examples", 0))
    train_n = int(protocol.get("train_examples", 0))
    require(train_n > 0 and validation_n > 0, "non-empty train and validation sets are required")
    require(protocol.get("primary_metric") == "multiple-choice accuracy", "primary metric changed")
    require(
        protocol.get("robustness_check") == "deterministic reversal of answer-choice order with label remapping",
        "choice-order robustness contract missing",
    )
    baseline_boundary = str(protocol.get("baseline_boundary", ""))
    require("not parameter matched" in baseline_boundary, "baseline limitation must remain explicit")
    require("strong pretrained baseline" in baseline_boundary, "strong-baseline limitation must remain explicit")
    claim_boundary = str(protocol.get("claim_boundary", ""))
    for phrase in ("external-data plumbing only", ">=5 seeds", "parameter-matched baseline", "strong pretrained baseline", "locked test-set", "independent reproduction"):
        require(phrase in claim_boundary, f"claim boundary missing: {phrase}")

    majority = metric_row(payload.get("majority_reference"), "majority_reference")
    require(isinstance(majority.get("majority_label"), int), "majority reference label missing")

    records = payload.get("records")
    require(isinstance(records, list) and len(records) == len(seeds), "seed record count mismatch")
    canonical_ids: list[str] | None = None
    for expected_seed, record in zip(seeds, records, strict=True):
        require(isinstance(record, dict), "seed record must be an object")
        require(record.get("seed") == expected_seed, "seed order mismatch")
        parameter_counts = record.get("parameter_counts")
        require(isinstance(parameter_counts, dict), "parameter counts missing")
        require(int(parameter_counts.get("hash_encoder_baseline", 0)) > 0, "baseline parameter count invalid")
        require(int(parameter_counts.get("lam_jepa_arc", 0)) > 0, "LAM parameter count invalid")
        metric_row(record.get("hash_encoder_baseline"), f"seed {expected_seed} hash baseline")
        metric_row(record.get("lam_jepa"), f"seed {expected_seed} LAM")
        metric_row(record.get("lam_jepa_reversed_choices"), f"seed {expected_seed} LAM reversed")
        raw = record.get("raw_predictions")
        require(isinstance(raw, dict), "raw predictions missing")
        baseline_ids = verify_prediction_rows(raw.get("hash_encoder_baseline"), validation_n, canonical_ids, "hash baseline")
        if canonical_ids is None:
            canonical_ids = baseline_ids
        verify_prediction_rows(raw.get("lam_jepa"), validation_n, canonical_ids, "LAM")
        verify_prediction_rows(raw.get("lam_jepa_reversed_choices"), validation_n, canonical_ids, "LAM reversed")

    report = {
        "status": "passed",
        "dataset": protocol["dataset"],
        "seeds": seeds,
        "train_examples": train_n,
        "validation_examples": validation_n,
        "validation_ids_verified": len(canonical_ids or []),
        "test_split_policy": protocol["test_split_policy"],
        "baseline_boundary": baseline_boundary,
        "claim_boundary": claim_boundary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
