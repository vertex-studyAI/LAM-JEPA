#!/usr/bin/env python3
"""Fail-closed, pre-outcome verifier for the LAM successor evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

PROTOCOL = "lam-arc-contextual-successor-v1-draft"
SCHEMA = "lam-arc-contextual-successor-v1-evidence-schema"
TEMPLATE = "lam-arc-contextual-successor-v1-run-receipt-template"
SEEDS = [11, 23, 37, 53, 71]
SYSTEMS = ["B0", "B1", "T1", "T2"]
DEPS = {
    "decision_rules": ("protocols/arc_successor_v1_decision_rules.json", "28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625"),
    "context_target": ("protocols/arc_successor_v1_context_target.json", "6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b"),
    "encoder": ("protocols/arc_successor_v1_encoder.json", "8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516"),
}
ROLES = [
    "raw_predictions", "metric_json", "representation_diagnostics",
    "optimizer_diagnostics", "stdout_stderr", "machine_readable_config",
    "split_manifest", "environment_lock", "host_metadata",
]
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), f"{path}: top level must be an object")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(contract: dict[str, Any], template: dict[str, Any], root: Path) -> dict[str, str]:
    require(contract.get("schema_id") == SCHEMA, "schema_id drift")
    require(contract.get("schema_version") == 1, "schema_version drift")
    require(contract.get("status") == "PREOUTCOME_CONTRACT_ONLY", "schema must remain pre-outcome")
    require(contract.get("protocol_id") == PROTOCOL, "protocol_id drift")
    require(contract.get("relationship_to_arc_v5") == "separate_versioned_study_no_rescue", "ARC-v5 relationship drift")
    require(contract.get("contract_does_not_authorize_execution") is True, "schema must not authorize execution")
    require(contract.get("contract_does_not_authorize_outcome_access") is True, "schema must not authorize outcome access")
    require(contract.get("allowed_primary_systems") == SYSTEMS, "system set/order drift")
    require(contract.get("frozen_primary_seeds") == SEEDS, "seed set/order drift")

    observed: dict[str, str] = {}
    frozen = contract.get("frozen_dependencies", {})
    for name, (path_text, expected) in DEPS.items():
        entry = frozen.get(name, {})
        require(entry.get("artifact") == path_text, f"{name} path drift")
        require(entry.get("sha256") == expected, f"{name} declared hash drift")
        path = root / path_text
        require(path.is_file(), f"missing dependency: {path_text}")
        actual = sha256(path)
        require(actual == expected, f"{name} byte drift: {actual}")
        observed[name] = actual

    encoder = frozen.get("encoder", {})
    require(encoder.get("repo_id") == "distilbert/distilroberta-base", "encoder repo drift")
    require(encoder.get("revision") == "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "encoder revision drift")
    require(encoder.get("checkpoint_sha256") == "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663", "checkpoint drift")

    require(contract.get("required_artifact_roles") == ROLES, "required artifact roles drift")
    artifacts = contract.get("artifact_contract", {})
    require(artifacts.get("each_role_requires") == ["path", "sha256", "bytes"], "artifact identity contract drift")
    require(artifacts.get("paths_must_be_relative") is True, "artifact paths must remain relative")
    require(artifacts.get("duplicate_roles_prohibited") is True, "duplicate roles must remain prohibited")
    require(artifacts.get("missing_required_role_fails_closed") is True, "missing artifacts must fail closed")

    auth = contract.get("authorization_contract", {})
    require(auth.get("complete_result_requires_all_true") is True, "complete results must require all authorization gates")
    claims = contract.get("claim_ledger_contract", {})
    for key in (
        "every_reported_number_requires_source_pointer",
        "secondary_metric_rescue_prohibited",
        "posthoc_seed_exclusion_prohibited",
        "failed_seed_omission_prohibited",
        "external_validation_claim_without_external_evidence_prohibited",
    ):
        require(claims.get(key) is True, f"claim-ledger guard missing: {key}")

    integrity = contract.get("integrity_rules", {})
    require(integrity.get("retain_failed_seeds") is True, "failed seeds must remain retained")
    require(integrity.get("posthoc_seed_exclusion") is False, "posthoc seed exclusion must remain false")
    require(integrity.get("heldout_threshold_tuning") is False, "held-out threshold tuning must remain false")
    require(integrity.get("architecture_shopping_after_failure") is False, "architecture shopping must remain false")
    require(integrity.get("secondary_metric_rescue") is False, "secondary rescue must remain false")
    require(integrity.get("old_locked_test_access") == "PROHIBITED", "old locked test must remain prohibited")
    require(integrity.get("schema_may_not_contain_scientific_outcomes") is True, "outcome-free schema rule missing")

    completion = contract.get("completion_contract", {})
    require(completion.get("allowed_statuses") == ["PREOUTCOME_ONLY", "BLOCKED", "FAILED_RUN_RETAINED", "COMPLETE_RESULT_PACKAGE"], "completion states drift")
    require(completion.get("complete_status") == "COMPLETE_RESULT_PACKAGE", "complete status drift")

    require(template.get("template_id") == TEMPLATE, "template_id drift")
    require(template.get("template_status") == "PREOUTCOME_ONLY", "template must remain pre-outcome")
    require(template.get("schema_artifact") == "protocols/arc_successor_v1_evidence_schema.json", "template schema path drift")
    require(all(section in template for section in contract.get("required_receipt_sections", [])), "template missing required section")

    identity = template.get("identity", {})
    require(identity.get("protocol_id") == PROTOCOL, "template protocol drift")
    require(identity.get("system_id") is None and identity.get("seed") is None and identity.get("source_sha") is None, "template must not preselect a run identity")
    require(identity.get("decision_rules_sha256") == DEPS["decision_rules"][1], "template decision-rule hash drift")
    require(identity.get("context_target_sha256") == DEPS["context_target"][1], "template context-target hash drift")
    require(identity.get("encoder_sha256") == DEPS["encoder"][1], "template encoder hash drift")

    for key in contract.get("authorization_contract", {}).get("required_fields", []):
        require(template.get("authorization", {}).get(key) is False, f"template authorization must remain false: {key}")
    require(template.get("artifacts") == [], "template must not contain result artifacts")
    require(template.get("claim_ledger") == [], "template must not contain scientific claims")

    data = template.get("data", {})
    require(data.get("dataset_id") is None, "confirmatory dataset must remain unresolved")
    require(data.get("dataset_bytes_sha256") is None, "confirmatory dataset bytes must remain unresolved")
    require(data.get("split_manifest_sha256") is None, "confirmatory split must remain unresolved")
    require(data.get("tokenizer_repo_id") == "distilbert/distilroberta-base", "tokenizer repo drift")
    require(data.get("tokenizer_revision") == "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "tokenizer revision drift")
    require(data.get("checkpoint_sha256") == "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663", "checkpoint drift")

    done = template.get("completion", {})
    require(done.get("status") == "PREOUTCOME_ONLY", "template completion status drift")
    require(done.get("receipt_sha256") is None, "template must not claim a future receipt hash")
    require(done.get("all_required_artifacts_present") is False, "template must not claim complete artifacts")
    require(done.get("all_frozen_seeds_retained") is False, "template must not claim completed seeds")
    require(done.get("claim_ledger_complete") is False, "template must not claim a complete ledger")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path("protocols/arc_successor_v1_evidence_schema.json"))
    parser.add_argument("--template", type=Path, default=Path("protocols/arc_successor_v1_run_receipt_template.json"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    contract_path = root / args.contract
    template_path = root / args.template
    contract, template = load(contract_path), load(template_path)
    observed = validate(contract, template, root)
    source_sha = os.environ.get("GITHUB_SHA")
    if source_sha is not None:
        require(bool(HEX40.fullmatch(source_sha)), "GITHUB_SHA must be lowercase 40-hex")

    report = {
        "status": "PASS_PREOUTCOME_EVIDENCE_SCHEMA",
        "source_sha": source_sha,
        "contract_sha256": sha256(contract_path),
        "template_sha256": sha256(template_path),
        "dependency_sha256": observed,
        "scientific_outcomes_read": False,
        "execution_authorized": False,
        "outcome_access_authorized": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
