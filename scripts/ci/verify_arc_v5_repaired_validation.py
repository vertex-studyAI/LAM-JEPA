from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_BASE = 20260808
CONDITIONS = ("legacy_ce", "repaired_v5_ce", "no_quantizer_ce", "repaired_v5_shuffled_labels")


def bootstrap_ci(values: Sequence[float], *, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = [float(statistics.fmean(values[rng.randrange(n)] for _ in range(n))) for _ in range(BOOTSTRAP_SAMPLES)]
    samples.sort()
    return samples[int(0.025 * (BOOTSTRAP_SAMPLES - 1))], samples[int(0.975 * (BOOTSTRAP_SAMPLES - 1))]


def close(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if result.get("protocol") != protocol:
        raise SystemExit("result package does not embed the exact frozen protocol")
    if protocol.get("status") != "FROZEN_BEFORE_VALIDATION_EXECUTION":
        raise SystemExit("protocol status is not frozen")
    if result.get("claim_boundary", {}).get("test_accessed") is not False:
        raise SystemExit("result package does not prove test_accessed=false")
    if result.get("claim_boundary", {}).get("research_complete") is not False:
        raise SystemExit("validation package must not claim research completion")

    validation = list(select_protocol_eligible_examples(load_arc_split(args.validation)).eligible)
    expected_ids = [example.item_id for example in validation]
    evidence = result["dataset_evidence"]
    if evidence["validation_eligible_rows"] != len(validation) != 0:
        raise SystemExit("validation row count mismatch")
    if evidence["validation_dataset_digest"] != dataset_digest(validation):
        raise SystemExit("validation dataset digest mismatch")
    if evidence["validation_id_digest"] != id_digest(validation):
        raise SystemExit("validation id digest mismatch")

    records = result["records"]
    seeds = [int(x) for x in protocol["training"]["seeds"]]
    recomputed_accuracy: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    repaired_support: list[tuple[int, float]] = []
    canonical_labels = None
    for condition in CONDITIONS:
        condition_records = records.get(condition)
        if not isinstance(condition_records, list) or [int(r["seed"]) for r in condition_records] != seeds:
            raise SystemExit(f"{condition}: seed records differ from frozen protocol")
        for record in condition_records:
            rows = record["rows"]
            if [row["id"] for row in rows] != expected_ids:
                raise SystemExit(f"{condition}/seed={record['seed']}: validation row identity/order mismatch")
            labels = [int(row["label"]) for row in rows]
            if canonical_labels is None:
                canonical_labels = labels
            elif labels != canonical_labels:
                raise SystemExit("validation labels differ across conditions/seeds")
            predictions = [int(row["prediction"]) for row in rows]
            accuracy = sum(int(p == y) for p, y in zip(predictions, labels, strict=True)) / len(labels)
            if not close(accuracy, float(record["accuracy"])):
                raise SystemExit(f"{condition}/seed={record['seed']}: reported accuracy mismatch")
            histogram = Counter(predictions)
            support = len(histogram)
            largest = max(histogram.values()) / len(predictions)
            if support != int(record["prediction_support"]) or not close(largest, float(record["largest_predicted_class_share"])):
                raise SystemExit(f"{condition}/seed={record['seed']}: collapse diagnostics mismatch")
            recomputed_accuracy[condition].append(float(accuracy))
            if condition == "repaired_v5_ce":
                repaired_support.append((support, largest))

    summary = result["summaries"]
    seed_offsets = {condition: BOOTSTRAP_BASE + i for i, condition in enumerate(CONDITIONS)}
    for condition in CONDITIONS:
        values = recomputed_accuracy[condition]
        low, high = bootstrap_ci(values, seed=seed_offsets[condition])
        if not close(statistics.fmean(values), float(summary[condition]["mean"])):
            raise SystemExit(f"{condition}: mean mismatch")
        if not close(low, float(summary[condition]["bootstrap_ci95_low"])) or not close(high, float(summary[condition]["bootstrap_ci95_high"])):
            raise SystemExit(f"{condition}: bootstrap interval mismatch")

    repaired_minus_legacy = [r - l for r, l in zip(recomputed_accuracy["repaired_v5_ce"], recomputed_accuracy["legacy_ce"], strict=True)]
    repaired_minus_noq = [r - n for r, n in zip(recomputed_accuracy["repaired_v5_ce"], recomputed_accuracy["no_quantizer_ce"], strict=True)]
    d1_low, _ = bootstrap_ci(repaired_minus_legacy, seed=BOOTSTRAP_BASE + 20)
    d2_low, _ = bootstrap_ci(repaired_minus_noq, seed=BOOTSTRAP_BASE + 21)
    neg_low, neg_high = bootstrap_ci(recomputed_accuracy["repaired_v5_shuffled_labels"], seed=seed_offsets["repaired_v5_shuffled_labels"])
    repaired_low, _ = bootstrap_ci(recomputed_accuracy["repaired_v5_ce"], seed=seed_offsets["repaired_v5_ce"])

    negative_control_valid = neg_high < 0.35
    collapse_rejected = all(support >= 2 and largest <= 0.95 for support, largest in repaired_support)
    generalization = bool(negative_control_valid and collapse_rejected and repaired_low > 0.25 and d1_low > 0.0)
    quantization_benefit = bool(d2_low > 0.0)
    rules = result["decision_rules"]
    expected_rules = {
        "negative_control_valid": negative_control_valid,
        "collapse_rejected": collapse_rejected,
        "generalization_supported_with_limitations": generalization,
        "quantization_benefit_supported": quantization_benefit,
    }
    if rules != expected_rules:
        raise SystemExit(f"decision-rule mismatch: expected {expected_rules}, got {rules}")

    if not negative_control_valid:
        expected_verdict = "INVALID_NEGATIVE_CONTROL"
    elif generalization:
        expected_verdict = "VALIDATION_GENERALIZATION_SUPPORTED_WITH_LIMITATIONS"
    else:
        expected_verdict = "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION"
    if result.get("verdict") != expected_verdict:
        raise SystemExit("verdict does not match independently recomputed rules")

    report = {
        "verdict": "ARC_V5_REPAIRED_VALIDATION_PACKAGE_VERIFIED",
        "scientific_verdict": expected_verdict,
        "seeds": seeds,
        "validation_rows": len(validation),
        "negative_control_valid": negative_control_valid,
        "collapse_rejected": collapse_rejected,
        "generalization_supported_with_limitations": generalization,
        "quantization_benefit_supported": quantization_benefit,
        "test_accessed": False,
        "research_complete": False,
        "independent_external_reproduction_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not negative_control_valid:
        raise SystemExit("invalid validation package: negative control failed")


if __name__ == "__main__":
    main()
