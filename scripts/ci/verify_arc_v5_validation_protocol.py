from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_COMMIT = "df249086e9171febaa77333a4c62888f35265c40"
TRAIN_SHA = "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb"
VALIDATION_SHA = "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text(encoding="utf-8"))

    require(p["protocol_id"] == "lam-jepa-arc-challenge-v5-repaired-validation", "protocol id drift")
    require(p["status"] == "FROZEN_BEFORE_VALIDATION_EXECUTION", "protocol not frozen")
    require(p["frozen_target_commit"] == TARGET_COMMIT, "target commit drift")
    require(p["repair_id"] == "arc-v5-stable-ema-residual-0.03125", "repair id drift")
    d = p["dataset"]
    require(d["train_sha256"] == TRAIN_SHA, "train digest drift")
    require(d["validation_sha256"] == VALIDATION_SHA, "validation digest drift")
    require(d["train_eligible_rows"] == 1117 and d["validation_eligible_rows"] == 295, "eligibility-count drift")
    require("must not" in d["test_split_policy"].lower(), "test split not fail-closed")
    t = p["training"]
    require(t == {
        "objective": "supervised_cross_entropy_only",
        "seeds": [1, 2, 3, 4, 5],
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "model_steps": 1,
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "train_rows": "all 1117 eligible rows",
        "validation_rows": "all 295 eligible rows",
        "no_early_stopping": True,
        "no_validation_model_selection": True,
        "no_validation_hyperparameter_selection": True,
    }, "training budget drift")
    require(set(p["conditions"]) == {"legacy_ce", "repaired_v5_ce", "no_quantizer_ce", "repaired_v5_shuffled_labels"}, "condition drift")
    rules = p["predeclared_decision_rules"]
    require("0.35" in rules["negative_control_valid"], "negative-control threshold drift")
    require("0.95" in rules["collapse_rejected"], "collapse threshold drift")
    require("above 0.25" in rules["generalization_supported_with_limitations"], "chance-bound rule missing")
    require("above 0" in rules["generalization_supported_with_limitations"], "paired legacy delta rule missing")
    require("above 0" in rules["quantization_benefit_supported"], "no-quantizer delta rule missing")
    bounds = p["claim_boundaries"]
    require(all(value is False for value in bounds.values()), "claim boundary weakened")
    require(p["after_validation"] == {
        "independent_result_reproduction_required": True,
        "no_further_v5_hyperparameter_tuning_on_validation": True,
        "confirmatory_test_requires_separate_explicit_authorization": True,
    }, "post-validation boundary drift")

    report = {
        "verdict": "ARC_V5_VALIDATION_PROTOCOL_FROZEN",
        "target_commit": TARGET_COMMIT,
        "train_sha256": TRAIN_SHA,
        "validation_sha256": VALIDATION_SHA,
        "validation_executed": False,
        "test_accessed": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
