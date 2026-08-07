from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind ARC matched-baseline smoke evidence to the frozen protocol-v2 capacity gate."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--base-verification", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    base = json.loads(args.base_verification.read_text(encoding="utf-8"))

    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v2", "wrong frozen protocol")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v2 is not frozen")
    require(base.get("verdict") == "CAPACITY_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY", "base verifier did not pass")
    require(base.get("locked_test_evaluated") is False, "development matched-baseline path touched locked test")

    matched_contract = ((protocol.get("models") or {}).get("matched_capacity_supervised_baseline") or {})
    lower = float(matched_contract.get("allowed_parameter_ratio_min", 0.0))
    upper = float(matched_contract.get("allowed_parameter_ratio_max", 0.0))
    require(lower == 0.99 and upper == 1.01, "protocol-v2 matched-capacity interval drift")

    result_protocol = results.get("protocol") or {}
    require(result_protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected benchmark dataset")
    require(result_protocol.get("train_validation_overlap") == 0, "train/validation leakage")
    require(
        result_protocol.get("test_split_policy") == "not downloaded or evaluated by this development command",
        "test isolation boundary changed",
    )
    require("gradient-active" in str(result_protocol.get("parameter_match_basis", "")), "wrong matching basis")

    lam_active = int(result_protocol.get("lam_gradient_active_parameters", 0))
    matched_active = int(result_protocol.get("matched_supervised_gradient_active_parameters", 0))
    matched_total = int(result_protocol.get("matched_supervised_trainable_parameters", 0))
    require(lam_active > 0 and matched_active > 0, "parameter counts must be positive")
    require(matched_active == matched_total, "matched supervised baseline contains inactive trainable padding")

    ratio = matched_active / lam_active
    require(lower <= ratio <= upper, f"protocol-v2 capacity ratio failed: {ratio:.9f} not in [{lower}, {upper}]")
    relative_gap = abs(matched_active - lam_active) / lam_active
    declared_gap = float(result_protocol.get("parameter_relative_gap", -1.0))
    require(abs(declared_gap - relative_gap) <= 1e-12, "declared parameter gap does not match exact counts")

    declared_tolerance = float(result_protocol.get("parameter_match_tolerance", -1.0))
    protocol_tolerance = max(1.0 - lower, upper - 1.0)
    require(
        0.0 < declared_tolerance <= protocol_tolerance + 1e-12,
        f"runner tolerance {declared_tolerance} is weaker than protocol-v2 tolerance {protocol_tolerance}",
    )

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "dataset": result_protocol["dataset"],
        "lam_gradient_active_parameters": lam_active,
        "matched_gradient_active_parameters": matched_active,
        "parameter_ratio": ratio,
        "parameter_relative_gap": relative_gap,
        "runner_tolerance": declared_tolerance,
        "protocol_ratio_interval": [lower, upper],
        "locked_test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
