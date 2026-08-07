from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed if the frozen ARC-Challenge scientific protocol drifts.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/manifests/arc_challenge.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-verification.json"))
    args = parser.parse_args()

    require(args.protocol.is_file(), f"protocol not found: {args.protocol}")
    require(args.dataset_manifest.is_file(), f"dataset manifest not found: {args.dataset_manifest}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))

    require(protocol.get("schema_version") == 1, "unsupported ARC protocol schema")
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v1", "unexpected protocol id")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol must remain frozen before test access")

    scope = protocol.get("claim_scope")
    require(isinstance(scope, dict), "claim scope missing")
    in_scope = scope.get("in_scope")
    out_of_scope = scope.get("out_of_scope")
    require(isinstance(in_scope, list) and in_scope, "in-scope claims missing")
    require(isinstance(out_of_scope, list) and out_of_scope, "out-of-scope claims missing")
    for forbidden_claim in ("student diagnosis", "adaptive tutoring effectiveness", "student-state modeling", "learning gains"):
        require(forbidden_claim in out_of_scope, f"ARC claim boundary weakened: {forbidden_claim} must remain out of scope")

    dataset = protocol.get("dataset")
    require(isinstance(dataset, dict), "dataset contract missing")
    require(dataset.get("name") == manifest.get("dataset") == "AI2 ARC-Challenge", "dataset identity mismatch")
    require(dataset.get("license") == manifest.get("license") == "CC-BY-SA-4.0", "dataset license mismatch")
    require(dataset.get("manifest") == "data/manifests/arc_challenge.json", "dataset manifest path changed")
    files = manifest.get("files")
    require(isinstance(files, dict), "dataset manifest files missing")
    for split in ("train", "validation", "test"):
        manifest_row = files.get(split)
        require(isinstance(manifest_row, dict), f"manifest split missing: {split}")
        expected_hash = manifest_row.get("sha256")
        require(isinstance(expected_hash, str) and len(expected_hash) == 64, f"invalid manifest hash: {split}")
        require(dataset.get(f"{split}_sha256") == expected_hash, f"protocol/manifest checksum drift: {split}")
    split_policy = dataset.get("split_policy")
    require(isinstance(split_policy, dict), "split policy missing")
    require("single confirmatory evaluation" in str(split_policy.get("test", "")), "test must remain confirmatory-only")
    require("must not influence model or hyperparameter selection" in str(dataset.get("leakage_rule", "")), "test-selection leakage rule missing")

    budget = protocol.get("training_budget")
    require(isinstance(budget, dict), "training budget missing")
    seeds = budget.get("training_seeds")
    require(seeds == [1, 2, 3, 4, 5], "confirmatory training seeds must remain exactly [1,2,3,4,5] in protocol v1")
    require(len(set(seeds)) == 5, "five unique seeds required")
    require(int(budget.get("epochs", 0)) > 0, "epochs must be positive")
    require(int(budget.get("batch_size", 0)) > 0, "batch size must be positive")
    require(budget.get("optimizer") == "AdamW", "optimizer drift")
    require(int(budget.get("model_steps", 0)) >= 1, "LAM-JEPA must exercise at least one planner step")
    require("identical rows, epochs, batch size, seeds" in str(budget.get("budget_rule", "")), "matched-budget rule missing")

    models = protocol.get("models")
    require(isinstance(models, dict), "model contract missing")
    matched = models.get("matched_capacity_supervised_baseline")
    require(isinstance(matched, dict), "matched-capacity baseline missing")
    require(float(matched.get("allowed_parameter_ratio_min", 0.0)) == 0.99, "matched baseline lower bound drift")
    require(float(matched.get("allowed_parameter_ratio_max", 0.0)) == 1.01, "matched baseline upper bound drift")
    accounting = str(matched.get("parameter_accounting", ""))
    require("gradient-active/student capacity" in accounting and "excluding EMA-only target parameters" in accounting, "parameter-accounting rule weakened")
    forbidden_components = set(matched.get("forbidden_components") or [])
    require({"EMA target encoder", "JEPA alignment loss", "latent-action planner", "sparse memory", "vector quantizer"} <= forbidden_components, "matched baseline gained forbidden LAM-JEPA mechanisms")

    pretrained = models.get("strong_pretrained_baseline")
    require(isinstance(pretrained, dict), "strong pretrained baseline missing")
    require(pretrained.get("model") == "microsoft/deberta-v3-xsmall", "pretrained baseline model drift")
    require(pretrained.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "pretrained baseline revision must remain pinned")
    require(pretrained.get("license") == "MIT", "pretrained baseline license drift")

    metrics = protocol.get("metrics")
    require(isinstance(metrics, dict), "metrics contract missing")
    require(metrics.get("primary") == "multiple-choice accuracy", "primary metric drift")
    require(float(metrics.get("practical_effect_threshold_absolute", 0.0)) == 0.02, "practical effect threshold drift")
    require(metrics.get("calibration_primary") == "Brier score", "calibration metric drift")
    require("95% bootstrap" in str(metrics.get("uncertainty", "")), "paired uncertainty contract missing")

    robustness = protocol.get("robustness")
    require(isinstance(robustness, dict), "robustness contract missing")
    require(float(robustness.get("maximum_allowed_lam_accuracy_drop", 1.0)) == 0.05, "robustness threshold drift")
    require("reverse answer choices" in str(robustness.get("choice_order", "")), "choice-order robustness missing")

    negative = protocol.get("negative_control")
    require(isinstance(negative, dict), "negative control missing")
    require(negative.get("type") == "deterministic training-label permutation", "negative control drift")
    require(negative.get("permutation_seed") == 20260807, "negative-control seed drift")
    require("never use confirmatory test labels" in str(negative.get("split", "")), "negative control must not consume test labels")

    ablations = protocol.get("ablations")
    require(isinstance(ablations, dict), "ablation contract missing")
    require(set(ablations.get("required") or []) == {"no_planner", "no_target"}, "required mechanism ablations drift")
    require("No student-state contribution claim" in str(ablations.get("student_state", "")), "student-state claim boundary weakened")

    gate = protocol.get("claim_gate")
    require(isinstance(gate, dict), "claim gate missing")
    require("at least 0.02" in str(gate.get("superiority", "")), "superiority effect threshold missing")
    require("excludes zero" in str(gate.get("superiority", "")), "superiority uncertainty gate missing")
    require("strongest trained non-JEPA baseline" in str(gate.get("strongest_baseline_rule", "")), "strongest-baseline gate missing")
    require("negative or inconclusive result" in str(gate.get("negative_results", "")), "negative-result retention rule missing")

    artifacts = protocol.get("artifact_contract")
    require(isinstance(artifacts, dict), "artifact contract missing")
    required_artifacts = set(artifacts.get("required") or [])
    for item in (
        "exact repository commit SHA",
        "exact dataset hashes",
        "exact model revisions",
        "all five seed records",
        "raw per-example probabilities and predictions",
        "parameter counts",
        "clean-checkout reproduction command",
    ):
        require(item in required_artifacts, f"required evidence removed: {item}")
    require("Lane 08" in str(artifacts.get("independent_qa", "")), "independent QA requirement missing")

    change_rule = str(protocol.get("protocol_change_rule", ""))
    require("new protocol version" in change_rule and "Never silently edit v1" in change_rule, "post-test protocol mutation guard missing")

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "dataset": dataset["name"],
        "training_seeds": seeds,
        "primary_metric": metrics["primary"],
        "matched_parameter_ratio": [matched["allowed_parameter_ratio_min"], matched["allowed_parameter_ratio_max"]],
        "strong_pretrained_baseline": {
            "model": pretrained["model"],
            "revision": pretrained["revision"],
            "license": pretrained["license"],
        },
        "confirmatory_test_policy": split_policy["test"],
        "claim_scope": in_scope,
        "out_of_scope": out_of_scope,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
