from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import verify_arc_protocol_v3_controls as strict

AGGREGATE_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(declared: object, expected: float, label: str, drifts: list[tuple[str, float]]) -> None:
    value = float(declared)
    drift = abs(value - expected)
    drifts.append((label, drift))
    require(math.isfinite(value), f"{label}: non-finite declared value")
    require(drift <= AGGREGATE_TOLERANCE, f"{label}: aggregate representation drift {drift} exceeds {AGGREGATE_TOLERANCE}")


def close_summary(declared: object, expected: dict[str, float | int], label: str, drifts: list[tuple[str, float]]) -> None:
    require(isinstance(declared, dict), f"{label}: summary missing")
    require(int(declared.get("n", -1)) == int(expected["n"]), f"{label}: n mismatch")
    close(declared.get("mean"), float(expected["mean"]), f"{label}/mean", drifts)
    close(declared.get("std"), float(expected["std"]), f"{label}/std", drifts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the full frozen ARC-v3 controls budget without rejecting harmless float32 aggregate serialization drift."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.results.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(raw)
    protocol = raw.get("protocol") or {}
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong result protocol")
    require(protocol.get("seeds") == [1, 2, 3, 4, 5], "full controls must use seeds 1..5")
    require(int(protocol.get("epochs", -1)) == 20, "full controls must use 20 epochs")
    require(int(protocol.get("batch_size", -1)) == 32, "full controls must use batch size 32")
    require(math.isclose(float(protocol.get("learning_rate", -1)), 0.0003, rel_tol=0.0, abs_tol=1e-12), "full controls learning rate drift")
    require(int(protocol.get("model_steps", -1)) == 1, "full controls model_steps drift")
    require(int((protocol.get("train_eligibility") or {}).get("used_rows", -1)) == 1117, "full controls must use all 1117 eligible train rows")
    require(int((protocol.get("validation_eligibility") or {}).get("used_rows", -1)) == 295, "full controls must use all 295 eligible validation rows")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked test boundary weakened")

    drifts: list[tuple[str, float]] = []
    recomputed_accuracy: dict[str, list[float]] = {}
    variants = raw.get("variants") or {}
    for variant in ("full", "no_planner", "no_target"):
        block = variants.get(variant) or {}
        records = block.get("records")
        require(isinstance(records, list) and len(records) == 5, f"{variant}: expected five seed records")
        values: list[float] = []
        for record in records:
            metrics, _, _ = strict.metrics_from_rows(record.get("predictions"))
            strict.verify_metrics(record.get("metrics"), metrics, f"{variant}/seed={record.get('seed')}")
            values.append(metrics["accuracy"])
        recomputed_accuracy[variant] = values
        expected_summary = strict.summarize(values)
        close_summary(block.get("accuracy"), expected_summary, f"{variant}/accuracy", drifts)
        normalized["variants"][variant]["accuracy"] = expected_summary

    normalized_paired: dict[str, dict[str, object]] = {}
    for offset, variant in enumerate(("no_planner", "no_target")):
        deltas = [
            full - ablated
            for full, ablated in zip(recomputed_accuracy["full"], recomputed_accuracy[variant], strict=True)
        ]
        declared = (raw.get("paired_effects") or {}).get(variant) or {}
        declared_deltas = declared.get("seed_level_full_minus_ablation")
        require(isinstance(declared_deltas, list) and len(declared_deltas) == len(deltas), f"{variant}: paired deltas missing")
        for index, (actual, expected) in enumerate(zip(declared_deltas, deltas, strict=True)):
            close(actual, expected, f"{variant}/delta[{index}]", drifts)
        mean_delta = float(statistics.fmean(deltas))
        std_delta = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
        ci_low, ci_high = strict.paired_bootstrap_ci(deltas, seed=20260807 + offset)
        close(declared.get("mean_full_minus_ablation"), mean_delta, f"{variant}/mean_delta", drifts)
        close(declared.get("std_paired_difference"), std_delta, f"{variant}/std_delta", drifts)
        close(declared.get("paired_bootstrap_ci95_low"), ci_low, f"{variant}/ci_low", drifts)
        close(declared.get("paired_bootstrap_ci95_high"), ci_high, f"{variant}/ci_high", drifts)
        numeric = mean_delta >= 0.01 and ci_low > 0.0
        require(declared.get("observed_mechanism_numeric_criterion_met") is numeric, f"{variant}: mechanism flag mismatch")
        normalized_paired[variant] = {
            "seed_level_full_minus_ablation": deltas,
            "mean_full_minus_ablation": mean_delta,
            "std_paired_difference": std_delta,
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "observed_mechanism_numeric_criterion_met": numeric,
        }
    normalized["paired_effects"] = normalized_paired

    negative = raw.get("negative_control") or {}
    negative_records = negative.get("records")
    require(isinstance(negative_records, list) and len(negative_records) == 5, "negative control must contain five seed records")
    negative_values: list[float] = []
    for record in negative_records:
        metrics, _, _ = strict.metrics_from_rows(record.get("predictions"))
        strict.verify_metrics(record.get("metrics"), metrics, f"negative/seed={record.get('seed')}")
        negative_values.append(metrics["accuracy"])
    negative_summary = strict.summarize(negative_values)
    close_summary(negative.get("accuracy"), negative_summary, "negative-control/accuracy", drifts)
    negative_pass = float(negative_summary["mean"]) <= 0.35
    require(negative.get("pass") is negative_pass, "negative-control pass flag mismatch")
    require(negative_pass, f"negative control failed: {negative_summary['mean']} > 0.35")
    normalized["negative_control"]["accuracy"] = negative_summary
    normalized["negative_control"]["pass"] = negative_pass

    normalized_path = args.report.with_name(args.report.stem + "-normalized-input.json")
    strict_report_path = args.report.with_name(args.report.stem + "-strict-report.json")
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/ci/verify_arc_protocol_v3_controls.py",
            "--results", str(normalized_path),
            "--protocol", str(args.protocol),
            "--train", str(args.train),
            "--validation", str(args.validation),
            "--report", str(strict_report_path),
        ],
        check=True,
    )
    strict_report = json.loads(strict_report_path.read_text(encoding="utf-8"))
    require(strict_report.get("verdict") == "PROTOCOL_V3_ARC_CONTROLS_EXECUTION_VERIFIED_ONLY", "strict structural verifier did not pass")
    require(strict_report.get("locked_test_evaluated") is False, "strict verifier observed locked-test access")

    max_drift_label, max_drift = max(drifts, key=lambda item: item[1], default=("none", 0.0))
    report = {
        **strict_report,
        "verdict": "PROTOCOL_V3_FULL_CONTROLS_VALIDATION_VERIFIED",
        "raw_results_preserved": True,
        "aggregate_normalization_only": True,
        "aggregate_tolerance": AGGREGATE_TOLERANCE,
        "maximum_observed_aggregate_drift": max_drift,
        "maximum_observed_aggregate_drift_field": max_drift_label,
        "seeds": [1, 2, 3, 4, 5],
        "epochs": 20,
        "batch_size": 32,
        "train_used_rows": 1117,
        "validation_used_rows": 295,
        "negative_control_pass": True,
        "locked_test_evaluated": False,
        "final_five_seed_20_epoch_protocol_executed": True,
        "mechanism_claim_authorized": False,
        "research_complete": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
