#!/usr/bin/env python3
"""Recompute bounded statistics for the frozen negative ARC study from retained evidence.

This verifier is intentionally outcome-closed: it reads only already-committed
validation/reproduction artifacts and never imports the model, downloads ARC data,
or accesses the locked confirmatory test.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIT = ROOT / "experiments/repro_wave_2026_08_13/independent_audit.json"
RECONCILED_AUDIT = ROOT / "audits/independent_artifact_audit_2026-08-13.json"
METRICS = ROOT / "experiments/repro_wave_2026_08_12/metrics.json"
EXPECTED_SCIENTIFIC_SHA = "760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb"
EXPECTED_SEEDS = [1, 2, 3, 4, 5]
EXPECTED_VALIDATION_ROWS = 295
EXPECTED_TRAIN_ROWS = 1117
TOL = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FROZEN_NEGATIVE_ARC_STAT_AUDIT_FAIL: {message}")


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def accuracy_to_exact_count(value: float, denominator: int) -> int:
    count = int(round(float(value) * denominator))
    require(0 <= count <= denominator, f"invalid derived count {count}/{denominator}")
    require(
        close(float(value), count / denominator, tol=5e-8),
        f"accuracy {value!r} is not consistent with an integer count over {denominator} rows",
    )
    return count


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    reconciled = json.loads(RECONCILED_AUDIT.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))["frozen_arc_protocol_v3_full_controls"]

    require(raw["scientific_head_sha"] == EXPECTED_SCIENTIFIC_SHA, "raw audit scientific SHA drift")
    require(reconciled["frozen_scientific_sha"] == EXPECTED_SCIENTIFIC_SHA, "reconciled audit scientific SHA drift")
    require(metrics["scientific_source_sha"] == EXPECTED_SCIENTIFIC_SHA, "metrics scientific SHA drift")

    protocol = raw["protocol"]
    verifier = reconciled["verifier"]
    require(protocol["seeds"] == EXPECTED_SEEDS, "raw audit seed list drift")
    require(verifier["seeds"] == EXPECTED_SEEDS, "verifier seed list drift")
    require(metrics["seeds"] == EXPECTED_SEEDS, "metrics seed list drift")
    require(protocol["eligible_train_rows"] == EXPECTED_TRAIN_ROWS, "raw train eligible-row drift")
    require(protocol["eligible_validation_rows"] == EXPECTED_VALIDATION_ROWS, "raw validation eligible-row drift")
    require(verifier["train_used_rows"] == verifier["train_eligible_rows"] == EXPECTED_TRAIN_ROWS, "not all eligible train rows were used")
    require(verifier["validation_used_rows"] == verifier["validation_eligible_rows"] == EXPECTED_VALIDATION_ROWS, "not all eligible validation rows were used")
    require(protocol["locked_test_evaluated"] is False, "raw audit says locked test was evaluated")
    require(verifier["locked_test_evaluated"] is False, "reconciled verifier says locked test was evaluated")
    require(metrics["locked_test_evaluated"] is False, "metrics say locked test was evaluated")
    require(verifier["verdict"] == "PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED", "verifier verdict drift")
    require(verifier["mechanism_claim_authorized"] is False, "mechanism claim was unexpectedly authorized")
    require(verifier["research_complete"] is False, "research_complete unexpectedly true")

    condition_map = {
        "full": "full_accuracy",
        "no_planner": "no_planner_accuracy",
        "no_target": "no_target_accuracy",
        "negative_control": "negative_control_accuracy",
    }
    per_seed_counts: dict[str, list[int]] = {}
    per_seed_accuracies: dict[str, list[float]] = {}

    for condition, raw_key in condition_map.items():
        values = [float(v) for v in raw["independent_raw_recomputation"][raw_key]["seed_values"]]
        require(len(values) == len(EXPECTED_SEEDS), f"{condition}: missing seed values")
        counts = [accuracy_to_exact_count(v, EXPECTED_VALIDATION_ROWS) for v in values]
        rational_values = [c / EXPECTED_VALIDATION_ROWS for c in counts]
        stats = summarize(rational_values)
        stored = reconciled["metrics"][condition]
        require(stored["n"] == len(EXPECTED_SEEDS), f"{condition}: stored n drift")
        require(close(stats["mean"], stored["mean_accuracy"]), f"{condition}: mean mismatch")
        require(close(stats["sample_sd"], stored["sample_sd"]), f"{condition}: sample SD mismatch")
        summary_metrics = metrics["summaries"][condition]
        require(close(stats["mean"], summary_metrics["mean"]), f"{condition}: metrics.json mean mismatch")
        require(close(stats["sample_sd"], summary_metrics["sample_sd"]), f"{condition}: metrics.json SD mismatch")
        per_seed_counts[condition] = counts
        per_seed_accuracies[condition] = rational_values

    planner_count_deltas = [a - b for a, b in zip(per_seed_counts["full"], per_seed_counts["no_planner"])]
    target_count_deltas = [a - b for a, b in zip(per_seed_counts["full"], per_seed_counts["no_target"])]
    planner_accuracy_deltas = [d / EXPECTED_VALIDATION_ROWS for d in planner_count_deltas]
    target_accuracy_deltas = [d / EXPECTED_VALIDATION_ROWS for d in target_count_deltas]

    planner_stats = summarize(planner_accuracy_deltas)
    target_stats = summarize(target_accuracy_deltas)
    require(close(planner_stats["mean"], reconciled["metrics"]["full_minus_no_planner"]["mean"]), "planner paired mean mismatch")
    require(close(planner_stats["sample_sd"], reconciled["metrics"]["full_minus_no_planner"]["sample_sd"]), "planner paired SD mismatch")
    require(close(target_stats["mean"], reconciled["metrics"]["full_minus_no_target"]["mean"]), "target paired mean mismatch")
    require(close(target_stats["sample_sd"], reconciled["metrics"]["full_minus_no_target"]["sample_sd"]), "target paired SD mismatch")

    planner_ci = reconciled["metrics"]["full_minus_no_planner"]["bootstrap_ci95"]
    target_ci = reconciled["metrics"]["full_minus_no_target"]["bootstrap_ci95"]
    require(planner_ci[0] <= 0.0 <= planner_ci[1], "planner interval unexpectedly excludes zero")
    require(target_ci[0] <= 0.0 <= target_ci[1], "target interval unexpectedly excludes zero")
    require(reconciled["metrics"]["full_minus_no_planner"]["criterion_met"] is False, "planner mechanism criterion unexpectedly met")
    require(reconciled["metrics"]["full_minus_no_target"]["criterion_met"] is False, "target mechanism criterion unexpectedly met")

    receipt = {
        "audit_id": "lam-jepa-frozen-negative-arc-statistical-closure-20260909",
        "scientific_source_sha": EXPECTED_SCIENTIFIC_SHA,
        "evidence_inputs": [
            str(RAW_AUDIT.relative_to(ROOT)),
            str(RECONCILED_AUDIT.relative_to(ROOT)),
            str(METRICS.relative_to(ROOT)),
        ],
        "dataset_semantics": {
            "benchmark": "AI2 ARC-Challenge multiple-choice validation",
            "train_eligible_rows": EXPECTED_TRAIN_ROWS,
            "train_used_rows": EXPECTED_TRAIN_ROWS,
            "validation_eligible_rows": EXPECTED_VALIDATION_ROWS,
            "validation_used_rows": EXPECTED_VALIDATION_ROWS,
            "all_eligible_rows_evaluated": True,
            "locked_confirmatory_test_evaluated": False,
        },
        "seeds": EXPECTED_SEEDS,
        "per_seed_correct_counts_over_295": per_seed_counts,
        "per_seed_accuracy": per_seed_accuracies,
        "paired_count_deltas": {
            "full_minus_no_planner": planner_count_deltas,
            "full_minus_no_target": target_count_deltas,
        },
        "paired_accuracy_statistics": {
            "full_minus_no_planner": planner_stats,
            "full_minus_no_target": target_stats,
        },
        "retained_bootstrap_intervals": {
            "full_minus_no_planner": planner_ci,
            "full_minus_no_target": target_ci,
        },
        "verdict": "NEGATIVE_OR_INCONCLUSIVE_STATISTICS_RECOMPUTED_FROM_RETAINED_EVIDENCE",
        "claim_boundary": {
            "arc_superiority_supported": False,
            "planner_mechanism_supported": False,
            "target_mechanism_supported": False,
            "locked_test_policy_preserved": True,
            "five_seed_uncertainty_is_descriptive_not_broad_significance": True,
            "no_new_outcome_access": True,
        },
    }

    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("FROZEN_NEGATIVE_ARC_STAT_AUDIT_PASS")


if __name__ == "__main__":
    main()
