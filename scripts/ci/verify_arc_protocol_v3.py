from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed if frozen ARC-Challenge protocol v3 drifts.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/manifests/arc_challenge.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-v3-verification.json"))
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))

    require(protocol.get("schema_version") == 3, "unsupported ARC protocol v3 schema")
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v3", "unexpected v3 protocol id")
    require(protocol.get("supersedes") == "lam-jepa-arc-challenge-v2", "v3 must explicitly supersede v2")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v3 must remain frozen before test access")
    reason = str(protocol.get("change_reason", ""))
    require("three or five choices" in reason and "No confirmatory test data" in reason, "v3 correction rationale/test boundary missing")

    dataset = protocol.get("dataset") or {}
    require(dataset.get("name") == manifest.get("dataset") == "AI2 ARC-Challenge", "dataset identity mismatch")
    require(dataset.get("license") == manifest.get("license") == "CC-BY-SA-4.0", "dataset license mismatch")
    files = manifest.get("files") or {}
    for split in ("train", "validation", "test"):
        expected = (files.get(split) or {}).get("sha256")
        require(isinstance(expected, str) and len(expected) == 64, f"manifest hash missing: {split}")
        require(dataset.get(f"{split}_sha256") == expected, f"protocol/manifest checksum drift: {split}")

    eligibility = dataset.get("eligibility") or {}
    require(eligibility.get("rule") == "retain a row if and only if len(choices) == 4", "eligibility rule drift")
    require(eligibility.get("required_choice_count") == 4, "required choice count drift")
    require("question structure only" in str(eligibility.get("decision_basis", "")), "eligibility must remain feature-only")
    for forbidden in ("answer labels", "predictions", "model outputs", "performance"):
        require(forbidden in str(eligibility.get("decision_basis", "")), f"eligibility decision boundary missing: {forbidden}")
    require("preserve original" in str(eligibility.get("ordering", "")), "eligibility ordering rule missing")
    require("excluded item IDs" in str(eligibility.get("exclusions", "")), "excluded-row evidence rule missing")
    require(eligibility.get("applies_to") == ["train", "validation", "test"], "eligibility must apply identically to all splits")

    split_policy = dataset.get("split_policy") or {}
    require("frozen eligibility rule" in str(split_policy.get("train", "")), "train eligibility policy missing")
    require("frozen eligibility rule" in str(split_policy.get("validation", "")), "validation eligibility policy missing")
    require("apply the already-frozen eligibility rule only after locked test access begins" in str(split_policy.get("test", "")), "test eligibility timing rule missing")
    leakage = str(dataset.get("leakage_rule", ""))
    require("Test labels" in leakage and "test eligibility counts" in leakage and "must not influence" in leakage, "pre-confirmatory test isolation rule weakened")

    scope = protocol.get("claim_scope") or {}
    in_scope = set(scope.get("in_scope") or [])
    out_of_scope = set(scope.get("out_of_scope") or [])
    require(any("exactly-four-choice" in item for item in in_scope), "v3 in-scope claim is not restricted to eligible rows")
    require("performance on ARC questions with other than four answer choices" in out_of_scope, "non-four-choice performance must remain out of scope")
    for claim in ("student diagnosis", "adaptive tutoring effectiveness", "student-state modeling", "learning gains"):
        require(claim in out_of_scope, f"claim boundary weakened: {claim}")

    budget = protocol.get("training_budget") or {}
    require(budget.get("training_seeds") == [1, 2, 3, 4, 5], "v3 seeds must remain exactly [1,2,3,4,5]")
    require(budget.get("epochs") == 20 and budget.get("batch_size") == 32, "v3 compute budget drift")
    require(budget.get("optimizer") == "AdamW", "optimizer drift")
    require(int(budget.get("model_steps", 0)) >= 1, "planner must be exercised")
    for key in ("train_examples", "validation_examples", "test_examples"):
        require("len(choices) == 4" in str(budget.get(key, "")), f"{key} is not bound to frozen eligibility")
    require("identical eligible rows" in str(budget.get("budget_rule", "")), "cross-model eligible-row pairing rule missing")

    models = protocol.get("models") or {}
    lam = models.get("lam_jepa") or {}
    require(lam.get("answer_head") == "four-choice ARC classifier", "LAM answer-head contract drift")
    matched = models.get("matched_capacity_supervised_baseline") or {}
    require("grad is not None" in str(matched.get("parameter_accounting", "")), "gradient-active matching missing")
    require(float(matched.get("allowed_parameter_ratio_min", 0.0)) == 0.99, "matched lower ratio drift")
    require(float(matched.get("allowed_parameter_ratio_max", 0.0)) == 1.01, "matched upper ratio drift")
    pretrained = models.get("strong_pretrained_baseline") or {}
    require(pretrained.get("model") == "microsoft/deberta-v3-xsmall", "strong baseline model drift")
    require(pretrained.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "strong baseline revision drift")
    require(pretrained.get("license") == "MIT", "strong baseline license drift")

    metrics = protocol.get("metrics") or {}
    require(metrics.get("primary") == "multiple-choice accuracy", "primary metric drift")
    require(float(metrics.get("practical_effect_threshold_absolute", 0.0)) == 0.02, "effect threshold drift")
    require(metrics.get("calibration_primary") == "Brier score", "calibration metric drift")
    require("95% bootstrap" in str(metrics.get("uncertainty", "")), "paired uncertainty contract missing")

    negative = protocol.get("negative_control") or {}
    require(negative.get("type") == "deterministic training-label permutation", "negative-control type drift")
    require(negative.get("permutation_seed") == 20260807, "negative-control seed drift")
    require("eligible validation only" in str(negative.get("split", "")), "negative control must use eligible validation")
    require("0.35" in str(negative.get("failure_rule", "")), "negative-control threshold drift")

    ablations = protocol.get("ablations") or {}
    require(set(ablations.get("required") or []) == {"no_planner", "no_target"}, "required ablations drift")

    gate = protocol.get("claim_gate") or {}
    require("at least 0.02" in str(gate.get("superiority", "")), "superiority threshold missing")
    require("excludes zero" in str(gate.get("superiority", "")), "superiority CI gate missing")
    require("strongest trained non-JEPA baseline" in str(gate.get("strongest_baseline_rule", "")), "strongest-baseline rule missing")
    require("negative or inconclusive result" in str(gate.get("negative_results", "")), "negative-result retention missing")

    artifacts = protocol.get("artifact_contract") or {}
    required_artifacts = set(artifacts.get("required") or [])
    for item in (
        "eligibility report for every accessed split",
        "eligible and excluded ordered-ID digests",
        "excluded item IDs and choice counts",
        "all five seed records",
        "raw per-example probabilities and predictions",
        "negative-control output",
        "clean-checkout reproduction command",
    ):
        require(item in required_artifacts, f"required v3 evidence removed: {item}")
    require("Lane 08" in str(artifacts.get("independent_qa", "")), "Lane 08 QA requirement missing")

    change_rule = str(protocol.get("protocol_change_rule", ""))
    require("V1 and V2 remain immutable" in change_rule, "prior protocol audit-trail protection missing")
    require("V3 supersedes V2" in change_rule and "new protocol version" in change_rule, "future protocol mutation guard missing")

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "supersedes": protocol["supersedes"],
        "dataset": dataset["name"],
        "eligibility_rule": eligibility["rule"],
        "required_choice_count": eligibility["required_choice_count"],
        "training_seeds": budget["training_seeds"],
        "strong_pretrained_baseline": {
            "model": pretrained["model"],
            "revision": pretrained["revision"],
            "license": pretrained["license"],
        },
        "test_policy": split_policy["test"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
