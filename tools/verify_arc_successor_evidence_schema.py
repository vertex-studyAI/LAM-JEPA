#!/usr/bin/env python3
"""Fail-closed verifier for the LAM successor retained-evidence contract.

This verifier is deliberately pre-outcome. It validates the evidence schema,
its immutable protocol dependencies, and the blank run-receipt template. It
never reads a held-out dataset, model output, or scientific metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL_ID = "lam-arc-contextual-successor-v1-draft"
EXPECTED_SCHEMA_ID = "lam-arc-contextual-successor-v1-evidence-schema"
EXPECTED_TEMPLATE_ID = "lam-arc-contextual-successor-v1-run-receipt-template"
EXPECTED_SEEDS = [11, 23, 37, 53, 71]
EXPECTED_SYSTEMS = ["B0", "B1", "T1", "T2"]
EXPECTED_DEPENDENCIES = {
    "decision_rules": (
        "protocols/arc_successor_v1_decision_rules.json",
        "28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625",
    ),
    "context_target": (
        "protocols/arc_successor_v1_context_target.json",
        "6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b",
    ),
    "encoder": (
        "protocols/arc_successor_v1_encoder.json",
        "8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516",
    ),
}
EXPECTED_REQUIRED_ROLES = [
    "raw_predictions",
    "metric_json",
    "representation_diagnostics",
    "optimizer_diagnostics",
    "stdout_stderr",
    "machine_readable_config",
    "split_manifest",
    "environment_lock",
    "host_metadata",
]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ContractError(f"{path}: top level must be an object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate_contract(contract: dict[str, Any], repo_root: Path) -> dict[str, str]:
    _require(contract.get("schema_id") == EXPECTED_SCHEMA_ID, "unexpected schema_id")
    _require(contract.get("schema_version") == 1, "schema_version must remain 1")
    _require(contract.get("status") == "PREOUTCOME_CONTRACT_ONLY", "schema must remain pre-outcome")
    _require(contract.get("protocol_id") == EXPECTED_PROTOCOL_ID, "protocol_id drift")
    _require(contract.get("relationship_to_arc_v5") == "separate_versioned_study_no_rescue", "ARC-v5 relationship drift")
    _require(contract.get("contract_does_not_authorize_execution") is True, "schema must not authorize execution")
    _require(contract.get("contract_does_not_authorize_outcome_access") is True, "schema must not authorize outcome access")
    _require(contract.get("allowed_primary_systems") == EXPECTED_SYSTEMS, "primary system set/order drift")
    _require(contract.get("frozen_primary_seeds") == EXPECTED_SEEDS, "frozen seed set/order drift")

    deps = contract.get("frozen_dependencies")
    _require(isinstance(deps, dict), "frozen_dependencies missing")
    observed: dict[str, str] = {}
    for name, (expected_path, expected_sha) in EXPECTED_DEPENDENCIES.items():
        entry = deps.get(name)
        _require(isinstance(entry, dict), f"dependency {name} missing")
        _require(entry.get("artifact") == expected_path, f"dependency path drift: {name}")
        _require(entry.get("sha256") == expected_sha, f"dependency declared SHA drift: {name}")
        path = repo_root / expected_path
        _require(path.is_file(), f"dependency file missing: {expected_path}")
        actual_sha = _sha256(path)
        _require(actual_sha == expected_sha, f"dependency byte drift: {name}: {actual_sha}")
        observed[name] = actual_sha

    encoder = deps["encoder"]
    _require(encoder.get("repo_id") == "distilbert/distilroberta-base", "encoder repo drift")
    _require(encoder.get("revision") == "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "encoder revision drift")
    _require(encoder.get("checkpoint_sha256") == "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663", "encoder checkpoint drift")

    _require(contract.get("required_artifact_roles") == EXPECTED_REQUIRED_ROLES, "required artifact roles drift")
    artifact_contract = contract.get("artifact_contract", {})
    _require(artifact_contract.get("each_role_requires") == ["path", "sha256", "bytes"], "artifact identity contract drift")
    _require(artifact_contract.get("paths_must_be_relative") is True, "artifact paths must remain relative")
    _require(artifact_contract.get("duplicate_roles_prohibited") is True, "duplicate artifact roles must remain prohibited")
    _require(artifact_contract.get("missing_required_role_fails_closed") is True, "missing artifacts must fail closed")

    auth = contract.get("authorization_contract", {})
    _require(auth.get("complete_result_requires_all_true") is True, "complete result must require all authorization gates")

    claims = contract.get("claim_ledger_contract", {})
    _require(claims.get("every_reported_number_requires_source_pointer") is True, "reported numbers must remain source-bound")
    _require(claims.get("secondary_metric_rescue_prohibited") is True, "secondary rescue must remain prohibited")
    _require(claims.get("posthoc_seed_exclusion_prohibited") is True, "posthoc seed exclusion must remain prohibited")
    _require(claims.get("failed_seed_omission_prohibited") is True, "failed seed omission must remain prohibited")
    _require(claims.get("external_validation_claim_without_external_evidence_prohibited") is True, "external-validation overclaim guard missing")

    integrity = contract.get("integrity_rules", {})
    _require(integrity.get("retain_failed_seeds") is True, "failed seeds must remain retained")
    _require(integrity.get("posthoc_seed_exclusion") is False, "posthoc seed exclusion must remain false")
    _require(integrity.get("heldout_threshold_tuning") is False, "held-out threshold tuning must remain false")
    _require(integrity.get("architecture_shopping_after_failure") is False, "architecture shopping must remain false")
    _require(integrity.get("secondary_metric_rescue") is False, "secondary metric rescue must remain false")
    _require(integrity.get("old_locked_test_access") == "PROHIBITED", "old locked test must remain prohibited")
    _require(integrity.get("schema_may_not_contain_scientific_outcomes") is True, "outcome-free schema rule missing")

    completion = contract.get("completion_contract", {})
    _require(completion.get("complete_status") == "COMPLETE_RESULT_PACKAGE", "complete status drift")
    _require(
        completion.get("allowed_statuses") == ["PREOUTCOME_ONLY", "BLOCKED", "FAILED_RUN_RETAINED", "COMPLETE_RESULT_PACKAGE"],
        "completion status set/order drift",
    )
    return observed


def validate_template(template: dict[str, Any], contract: dict[str, Any]) -> None:
    _require(template.get("template_id") == EXPECTED_TEMPLATE_ID, "unexpected template_id")
    _require(template.get("template_status") == "PREOUTCOME_ONLY", "template must remain pre-outcome")
    _require(template.get("schema_artifact") == "protocols/arc_successor_v1_evidence_schema.json", "template schema path drift")

    required_sections = contract.get("required_receipt_sections", [])
    _require(all(section in template for section in required_sections), "template missing required receipt section")

    identity = template.get("identity", {})
    _require(identity.get("protocol_id") == EXPECTED_PROTOCOL_ID, "template protocol drift")
    _require(identity.get("system_id") is None, "pre-outcome template must not pick a system")
    _require(identity.get("seed") is None, "pre-outcome template must not pick a seed")
    _require(identity.get("source_sha") is None, "pre-outcome template must not bind a future execution source")
    _require(identity.get("decision_rules_sha256") == EXPECTED_DEPENDENCIES["decision_rules"][1], "template decision-rule hash drift")
    _require(identity.get("context_target_sha256") == EXPECTED_DEPENDENCIES["context_target"][1], "template context-target hash drift")
    _require(identity.get("encoder_sha256") == EXPECTED_DEPENDENCIES["encoder"][1], "template encoder hash drift")

    authorization = template.get("authorization", {})
    for key in contract.get("authorization_contract", {}).get("required_fields", []):
        _require(authorization.get(key) is False, f"pre-outcome authorization field must remain false: {key}")

    _require(template.get("artifacts") == [], "pre-outcome template must not contain retained result artifacts")
    _require(template.get("claim_ledger") == [], "pre-outcome template must not contain scientific claims")
    completion = template.get("completion", {})
    _require(completion.get("status") == "PREOUTCOME_ONLY", "template completion status drift")
    _require(completion.get("receipt_sha256") is None, "template must not claim a future receipt hash")
    _require(completion.get("all_required_artifacts_present") is False, "template must not claim complete artifacts")
    _require(completion.get("all_frozen_seeds_retained") is False, "template must not claim completed seed retention")
    _require(completion.get("claim_ledger_complete") is False, "template must not claim complete claim ledger")

    data = template.get("data", {})
    _require(data.get("dataset_id") is None, "confirmatory dataset must remain unresolved in template")
    _require(data.get("dataset_bytes_sha256") is None, "confirmatory dataset bytes must remain unresolved in template")
    _require(data.get("split_manifest_sha256") is None, "confirmatory split must remain unresolved in template")
    _require(data.get("tokenizer_repo_id") == "distilbert/distilroberta-base", "tokenizer repo drift")
    _require(data.get("tokenizer_revision") == "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "tokenizer revision drift")
    _require(data.get("checkpoint_sha256") == "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663", "checkpoint drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("protocols/arc_successor_v1_evidence_schema.json"))
    parser.add_argument("--template", type=Path, default=Path("protocols/arc_successor_v1_run_receipt_template.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    contract_path = (root / args.contract).resolve() if not args.contract.is_absolute() else args.contract
    template_path = (root / args.template).resolve() if not args.template.is_absolute() else args.template
    contract = _load(contract_path)
    template = _load(template_path)
    observed = validate_contract(contract, root)
    validate_template(template, contract)

    source_sha = os.environ.get("GITHUB_SHA")
    if source_sha is not None:
        _require(bool(HEX40.fullmatch(source_sha)), "GITHUB_SHA must be a 40-character lowercase commit SHA")

    report = {
        "status": "PASS_PREOUTCOME_EVIDENCE_SCHEMA",
        "source_sha": source_sha,
        "contract_sha256": _sha256(contract_path),
        "template_sha256": _sha256(template_path),
        "dependency_sha256": observed,
        "scientific_outcomes_read": false,
        "execution_authorized": false,
        "outcome_access_authorized": false
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
