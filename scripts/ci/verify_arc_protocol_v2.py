from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed if the frozen ARC-Challenge v2 protocol drifts.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/manifests/arc_challenge.json"))
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-protocol-v2-verification.json"))
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))

    require(protocol.get("schema_version") == 2, "unsupported ARC protocol v2 schema")
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v2", "unexpected v2 protocol id")
    require(protocol.get("supersedes") == "lam-jepa-arc-challenge-v1", "v2 must explicitly supersede v1")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v2 must remain frozen before test access")
    require("auxiliary heads" in str(protocol.get("change_reason", "")), "v2 QA correction rationale missing")

    dataset = protocol.get("dataset") or {}
    require(dataset.get("name") == manifest.get("dataset") == "AI2 ARC-Challenge", "dataset identity mismatch")
    require(dataset.get("license") == manifest.get("license") == "CC-BY-SA-4.0", "dataset license mismatch")
    files = manifest.get("files") or {}
    for split in ("train", "validation", "test"):
        expected = (files.get(split) or {}).get("sha256")
        require(isinstance(expected, str) and len(expected) == 64, f"manifest hash missing: {split}")
        require(dataset.get(f"{split}_sha256") == expected, f"protocol/manifest checksum drift: {split}")
    split_policy = dataset.get("split_policy") or {}
    require("single confirmatory evaluation" in str(split_policy.get("test", "")), "test must remain confirmatory-only")
    require("must not influence model or hyperparameter selection" in str(dataset.get("leakage_rule", "")), "leakage rule missing")

    scope = protocol.get("claim_scope") or {}
    out_of_scope = set(scope.get("out_of_scope") or [])
    for claim in ("student diagnosis", "adaptive tutoring effectiveness", "student-state modeling", "learning gains"):
        require(claim in out_of_scope, f"claim boundary weakened: {claim}")

    budget = protocol.get("training_budget") or {}
    require(budget.get("training_seeds") == [1, 2, 3, 4, 5], "v2 seeds must remain exactly [1,2,3,4,5]")
    require(budget.get("optimizer") == "AdamW", "optimizer drift")
    require(int(budget.get("epochs", 0)) > 0 and int(budget.get("batch_size", 0)) > 0, "training budget invalid")
    require(int(budget.get("model_steps", 0)) >= 1, "planner must be exercised")

    models = protocol.get("models") or {}
    matched = models.get("matched_capacity_supervised_baseline") or {}
    accounting = str(matched.get("parameter_accounting", ""))
    require("parameter.grad is not None" in accounting, "v2 must use empirical gradient-active accounting")
    require("_lam_arc_loss backward" in accounting, "v2 must bind accounting to the exact ARC objective")
    require("auxiliary heads disconnected from the ARC loss" in accounting, "v2 must exclude ARC-disconnected auxiliary heads")
    require(float(matched.get("allowed_parameter_ratio_min", 0.0)) == 0.99, "matched lower ratio drift")
    require(float(matched.get("allowed_parameter_ratio_max", 0.0)) == 1.01, "matched upper ratio drift")
    require("gradient-active" in str(matched.get("forward_path_requirement", "")), "baseline forward-path gate missing")
    forbidden = set(matched.get("forbidden_components") or [])
    require({"EMA target encoder", "JEPA alignment loss", "latent-action planner", "sparse memory", "vector quantizer"} <= forbidden, "forbidden baseline mechanisms drift")

    pretrained = models.get("strong_pretrained_baseline") or {}
    require(pretrained.get("model") == "microsoft/deberta-v3-xsmall", "pretrained baseline model drift")
    require(pretrained.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "pretrained baseline revision drift")
    require(pretrained.get("license") == "MIT", "pretrained baseline license drift")

    metrics = protocol.get("metrics") or {}
    require(metrics.get("primary") == "multiple-choice accuracy", "primary metric drift")
    require(float(metrics.get("practical_effect_threshold_absolute", 0.0)) == 0.02, "effect threshold drift")
    require(metrics.get("calibration_primary") == "Brier score", "calibration metric drift")
    require("95% bootstrap" in str(metrics.get("uncertainty", "")), "paired uncertainty contract missing")

    negative = protocol.get("negative_control") or {}
    require(negative.get("permutation_seed") == 20260807, "negative-control seed drift")
    require("never use confirmatory test labels" in str(negative.get("split", "")), "negative control test isolation missing")

    ablations = protocol.get("ablations") or {}
    require(set(ablations.get("required") or []) == {"no_planner", "no_target"}, "required ablations drift")

    gate = protocol.get("claim_gate") or {}
    require("at least 0.02" in str(gate.get("superiority", "")), "superiority threshold missing")
    require("excludes zero" in str(gate.get("superiority", "")), "superiority CI gate missing")
    require("strongest trained non-JEPA baseline" in str(gate.get("strongest_baseline_rule", "")), "strongest-baseline rule missing")
    require("negative or inconclusive result" in str(gate.get("negative_results", "")), "negative-result retention missing")

    required_artifacts = set((protocol.get("artifact_contract") or {}).get("required") or [])
    for item in (
        "all five seed records",
        "raw per-example probabilities and predictions",
        "LAM-JEPA ARC gradient-active parameter report",
        "matched-baseline gradient-active parameter report",
        "parameter-count ratio evidence",
        "clean-checkout reproduction command",
    ):
        require(item in required_artifacts, f"required v2 evidence removed: {item}")
    require("Lane 08" in str((protocol.get("artifact_contract") or {}).get("independent_qa", "")), "Lane 08 QA missing")

    change_rule = str(protocol.get("protocol_change_rule", ""))
    require("V1 remains immutable" in change_rule, "v1 audit-trail protection missing")
    require("new protocol version" in change_rule and "Never silently edit V1 or V2" in change_rule, "protocol mutation guard missing")

    report = {
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "supersedes": protocol["supersedes"],
        "dataset": dataset["name"],
        "training_seeds": budget["training_seeds"],
        "parameter_accounting": accounting,
        "matched_parameter_ratio": [matched["allowed_parameter_ratio_min"], matched["allowed_parameter_ratio_max"]],
        "confirmatory_test_policy": split_policy["test"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
