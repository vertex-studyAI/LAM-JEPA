from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MODEL = "microsoft/deberta-v3-xsmall"
EXPECTED_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed verifier for frozen ARC protocol v3.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    v3 = read(args.protocol)
    v2 = read(args.v2)
    manifest = read(args.dataset_manifest)

    require(v3.get("schema_version") == 3, "protocol schema_version must be 3")
    require(v3.get("protocol_id") == "lam-jepa-arc-challenge-v3", "unexpected protocol id")
    require(v3.get("supersedes") == "lam-jepa-arc-challenge-v2", "v3 must explicitly supersede v2")
    require(v3.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v3 must remain frozen before test")
    require(v2.get("protocol_id") == "lam-jepa-arc-challenge-v2", "unexpected v2 id")
    require(v2.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "v2 freeze history changed")

    # Scientific question and public claim gate cannot drift under the cardinality repair.
    for key in ("scientific_question", "claim_scope", "claim_gate"):
        require(v3.get(key) == v2.get(key), f"unexpected v2→v3 scientific drift: {key}")

    # Every executable training-budget value remains identical. Only the prose is allowed to
    # clarify that the declared batch size counts questions while candidate inputs are flattened.
    budget_v2 = v2.get("training_budget", {})
    budget_v3 = v3.get("training_budget", {})
    budget_fields = (
        "training_seeds",
        "epochs",
        "batch_size",
        "optimizer",
        "lam_jepa_learning_rate",
        "matched_baseline_learning_rate",
        "pretrained_baseline_learning_rate",
        "model_steps",
        "train_examples",
        "validation_examples",
        "test_examples",
    )
    for key in budget_fields:
        require(budget_v3.get(key) == budget_v2.get(key), f"training-budget value drift: {key}")
    budget_rule = str(budget_v3.get("budget_rule", ""))
    require("question-level batch size" in budget_rule, "v3 batch-size semantics must be question-level")
    require("identical rows" in budget_rule and "same rows" in budget_rule, "v3 budget rule must retain cross-model row equality")

    # Dataset identity and hashes are immutable; only row-policy/cardinality evidence is added.
    for key in ("name", "manifest", "license", "train_sha256", "validation_sha256", "test_sha256", "split_policy", "leakage_rule"):
        require(v3.get("dataset", {}).get(key) == v2.get("dataset", {}).get(key), f"dataset contract drift: {key}")
    require(v3["dataset"].get("row_policy") == "Use every manifest-declared row. Do not filter by answer-choice cardinality. The scorer must support the number of choices present in each item.", "full-row policy missing")
    card = v3["dataset"].get("pre_test_cardinality_evidence")
    require(isinstance(card, dict), "pre-test cardinality evidence missing")
    require(card.get("test_split_accessed") is False, "cardinality repair must not use test")
    require(card.get("train_rows") == manifest["files"]["train"]["rows"] == 1119, "train row count mismatch")
    require(card.get("validation_rows") == manifest["files"]["validation"]["rows"] == 299, "validation row count mismatch")
    require(card.get("train_choice_count_distribution") == {"3": 1, "4": 1117, "5": 1}, "train cardinality evidence drift")
    require(card.get("validation_choice_count_distribution") == {"3": 3, "4": 295, "5": 1}, "validation cardinality evidence drift")
    require(card.get("audit_artifact_id") == 9000175495, "cardinality audit artifact id drift")
    require(card.get("audit_artifact_sha256") == "8c718f4d3b1dcd21fc4e9b88662cd89b1151fdb8986efadaca92833601243671", "cardinality artifact digest drift")

    interface = v3.get("answer_interface")
    require(isinstance(interface, dict), "variable-choice answer interface missing")
    require(interface.get("type") == "variable-choice shared candidate scorer", "unexpected answer-interface type")
    require("actual number of choices" in interface.get("normalization", ""), "per-question candidate normalization missing")
    require("Do not encode candidate position" in interface.get("candidate_input", ""), "position-bias prohibition missing")
    require("question is the unit" in interface.get("batching", "").lower(), "question-level batching contract missing")
    require("row dropped" in interface.get("failure_rule", ""), "full-row failure rule missing")

    models = v3.get("models", {})
    require(models.get("lam_jepa", {}).get("answer_head") == "shared scalar candidate scorer over the LAM-JEPA representation", "LAM variable-choice scorer missing")
    require(models.get("lam_jepa", {}).get("planner_steps") == v2.get("models", {}).get("lam_jepa", {}).get("planner_steps"), "LAM planner-step contract changed")
    matched = models.get("matched_capacity_supervised_baseline", {})
    require("variable-choice" in matched.get("parameter_accounting", ""), "matched baseline is not bound to v3 objective")
    require(matched.get("allowed_parameter_ratio_min") == 0.99 and matched.get("allowed_parameter_ratio_max") == 1.01, "matched parameter tolerance changed")
    require(matched.get("forbidden_components") == v2.get("models", {}).get("matched_capacity_supervised_baseline", {}).get("forbidden_components"), "matched forbidden-component contract changed")
    pretrained = models.get("strong_pretrained_baseline", {})
    require(pretrained.get("model") == EXPECTED_MODEL, "strong pretrained model changed")
    require(pretrained.get("revision") == EXPECTED_REVISION, "strong pretrained revision changed")
    require(pretrained.get("license") == "MIT", "strong pretrained license changed")
    require(pretrained.get("role") == v2.get("models", {}).get("strong_pretrained_baseline", {}).get("role"), "pretrained role changed")
    require(pretrained.get("parameter_matching") == v2.get("models", {}).get("strong_pretrained_baseline", {}).get("parameter_matching"), "pretrained parameter boundary changed")
    require("preserves every question" in pretrained.get("variable_choice_rule", ""), "pretrained full-row rule missing")

    negative = v3.get("negative_control", {})
    require(negative.get("type") == "deterministic training-label permutation within choice-cardinality strata", "v3 negative-control type invalid")
    require(negative.get("permutation_seed") == v2.get("negative_control", {}).get("permutation_seed") == 20260807, "negative-control seed changed")
    require(negative.get("split") == v2.get("negative_control", {}).get("split"), "negative-control split changed")
    require("same number of choices" in negative.get("permutation_rule", ""), "cardinality-safe permutation rule missing")
    require(negative.get("failure_rule") == v2.get("negative_control", {}).get("failure_rule"), "negative-control stop rule changed")

    require(v3.get("ablations") == v2.get("ablations"), "ablation contract changed during cardinality repair")
    require(v3.get("robustness", {}).get("maximum_allowed_lam_accuracy_drop") == v2.get("robustness", {}).get("maximum_allowed_lam_accuracy_drop"), "robustness threshold changed")
    require(v3.get("robustness", {}).get("failure_rule") == v2.get("robustness", {}).get("failure_rule"), "robustness failure rule changed")
    require(v3.get("metrics", {}).get("primary") == v2.get("metrics", {}).get("primary"), "primary metric changed")
    require(v3.get("metrics", {}).get("practical_effect_threshold_absolute") == v2.get("metrics", {}).get("practical_effect_threshold_absolute") == 0.02, "practical effect threshold changed")
    require(v3.get("metrics", {}).get("uncertainty") == v2.get("metrics", {}).get("uncertainty"), "uncertainty rule changed")
    require(v3.get("metrics", {}).get("calibration_descriptive") == v2.get("metrics", {}).get("calibration_descriptive"), "descriptive calibration metrics changed")
    require("actual candidate set" in v3.get("metrics", {}).get("calibration_primary", ""), "variable-cardinality Brier semantics missing")

    artifact_required = set(v3.get("artifact_contract", {}).get("required", []))
    require("proof that every manifest-declared split row was evaluated" in artifact_required, "full-row artifact evidence missing")
    require("raw per-example candidate probabilities and predictions with actual choice counts" in artifact_required, "variable-choice raw artifact requirement missing")
    require(v3.get("artifact_contract", {}).get("independent_qa") == v2.get("artifact_contract", {}).get("independent_qa"), "independent QA gate changed")
    require("V3 supersedes V2 before confirmatory test access" in v3.get("protocol_change_rule", ""), "v3 audit-history rule missing")

    # Manifest hashes must equal the frozen protocol hashes exactly.
    files = manifest.get("files", {})
    require(files["train"]["sha256"] == v3["dataset"]["train_sha256"], "train hash mismatch")
    require(files["validation"]["sha256"] == v3["dataset"]["validation_sha256"], "validation hash mismatch")
    require(files["test"]["sha256"] == v3["dataset"]["test_sha256"], "test hash mismatch")

    report = {
        "status": "passed",
        "protocol_id": v3["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol),
        "supersedes": v3["supersedes"],
        "test_accessed_for_repair": False,
        "full_split_row_policy": True,
        "variable_choice_interface_frozen": True,
        "strong_pretrained_model_unchanged": True,
        "training_budget_values_unchanged": True,
        "batch_size_semantics_clarified_to_questions": True,
        "claim_gate_unchanged": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
