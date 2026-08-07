from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_MODULE_TYPES = {
    "LAMJEPA",
    "EMAQuantizer",
    "LatentActionModel",
    "SparseMemory",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the protocol-v2 matched-capacity ARC smoke evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    records = payload.get("records") or []

    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v2", "wrong ARC protocol id")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "wrong ARC dataset")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    require(protocol.get("seeds") == args.expected_seeds, "seed set drift")
    require(int(protocol.get("model_steps", 0)) >= 1, "LAM planner was not exercised")
    require(protocol.get("capacity_accounting") == "gradient-active parameters after exact ARC loss backward", "capacity accounting drift")
    require(protocol.get("allowed_parameter_ratio") == [0.99, 1.01], "parameter-ratio gate drift")
    require(protocol.get("test_split_policy") == "not loaded or accessed by this smoke", "test split access boundary changed")
    require("No performance" in str(protocol.get("claim_boundary", "")), "development-smoke claim boundary missing")
    require(len(records) == len(args.expected_seeds), "unexpected number of seed records")

    verified: list[dict[str, object]] = []
    for expected_seed, record in zip(args.expected_seeds, records, strict=True):
        require(record.get("seed") == expected_seed, f"seed record order mismatch: expected {expected_seed}")
        lam = record.get("lam_parameter_accounting") or {}
        matched = record.get("matched_baseline_parameter_accounting") or {}
        target = int(lam.get("gradient_active_parameters", 0))
        lam_trainable = int(lam.get("requires_grad_parameters", 0))
        lam_inactive = int(lam.get("gradient_inactive_parameters", -1))
        matched_active = int(matched.get("gradient_active_parameters", 0))
        matched_trainable = int(matched.get("trainable_parameters", 0))
        ratio = float(matched.get("parameter_ratio", 0.0))

        require(target > 0, f"seed {expected_seed}: invalid LAM gradient-active target")
        require(lam_trainable > target, f"seed {expected_seed}: QA distinction disappeared; LAM active must be < requires-grad")
        require(lam_trainable - target == lam_inactive, f"seed {expected_seed}: LAM inactive count mismatch")
        require(matched_active == matched_trainable, f"seed {expected_seed}: baseline contains ARC-inactive parameters")
        require(matched.get("inactive_parameter_names") == [], f"seed {expected_seed}: baseline inactive parameter list not empty")
        require(abs(ratio - (matched_active / target)) < 1e-12, f"seed {expected_seed}: parameter ratio is not derived from counts")
        require(0.99 <= ratio <= 1.01, f"seed {expected_seed}: matched baseline outside 0.99–1.01 capacity gate")
        require(abs(1.0 - ratio) <= 0.001, f"seed {expected_seed}: constructor did not choose a near-exact match")

        module_types = set(matched.get("module_types") or [])
        require(not (module_types & FORBIDDEN_MODULE_TYPES), f"seed {expected_seed}: forbidden LAM mechanism in baseline: {sorted(module_types & FORBIDDEN_MODULE_TYPES)}")
        require("MatchedCapacityARCClassifier" in module_types, f"seed {expected_seed}: expected matched baseline class missing")

        raw = record.get("raw_predictions") or {}
        matched_rows = raw.get("matched_baseline") or []
        lam_rows = raw.get("lam_jepa") or []
        require(matched_rows and len(matched_rows) == len(lam_rows), f"seed {expected_seed}: raw prediction lengths differ")
        require([row.get("id") for row in matched_rows] == [row.get("id") for row in lam_rows], f"seed {expected_seed}: raw prediction IDs are not paired")
        for model_name in ("matched_baseline", "lam_jepa"):
            metrics = record.get(model_name) or {}
            accuracy = float(metrics.get("accuracy", -1.0))
            brier = float(metrics.get("brier", -1.0))
            require(0.0 <= accuracy <= 1.0, f"seed {expected_seed}: invalid {model_name} accuracy")
            require(brier >= 0.0, f"seed {expected_seed}: invalid {model_name} Brier score")

        verified.append(
            {
                "seed": expected_seed,
                "lam_gradient_active_parameters": target,
                "lam_requires_grad_parameters": lam_trainable,
                "matched_gradient_active_parameters": matched_active,
                "parameter_ratio": ratio,
                "hidden_width": int(matched.get("hidden_width", 0)),
                "matched_accuracy": float(record["matched_baseline"]["accuracy"]),
                "lam_accuracy": float(record["lam_jepa"]["accuracy"]),
            }
        )

    targets = {row["lam_gradient_active_parameters"] for row in verified}
    require(len(targets) == 1, "LAM gradient-active target changed across smoke seeds")

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "test_split_policy": protocol["test_split_policy"],
        "verified_records": verified,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
