#!/usr/bin/env python3
"""Verify the LAM-JEPA successor pre-outcome execution-lock contract.

This verifier is intentionally outcome-blind. It reads only protocol/control-plane
JSON and source text. It never imports model, dataset, or scientific runner code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "protocols" / "arc_successor_v1_execution_contract.json"
PROTOCOL = ROOT / "protocols" / "arc_successor_v1_draft.json"
EVIDENCE_SCHEMA = ROOT / "protocols" / "arc_successor_v1_evidence_schema.json"
ENTRYPOINT = ROOT / "tools" / "arc_successor_locked_entrypoint.py"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SYSTEMS = ["B0", "B1", "T1", "T2"]
EXPECTED_SEEDS = [11, 23, 37, 53, 71]
EXPECTED_CURRENT_BLOCKERS = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}
EXPECTED_SOURCE_FIELDS = {
    "source_sha",
    "source_tree_sha",
    "protocol_sha256",
    "decision_rules_sha256",
    "context_target_sha256",
    "encoder_sha256",
    "evidence_schema_sha256",
    "execution_contract_sha256",
}
EXPECTED_ENVIRONMENT_FIELDS = {
    "environment_lock_sha256",
    "python_version",
    "torch_version",
    "transformers_version",
    "tokenizers_version",
    "numpy_version",
    "device_class",
    "precision",
    "host_metadata_sha256",
}
EXPECTED_MANIFEST_FIELDS = {
    "exact_reproduce_command",
    "implementation_sha256",
    "optimizer_contract_sha256",
    "budget_contract_sha256",
    "analysis_sha256",
    "dataset_snapshot_sha256",
    "split_manifest_sha256",
    "authorization_receipt_sha256",
}
EXPECTED_DEPENDENCY_HASHES = {
    "decision_rules": "28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625",
    "context_target": "6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b",
    "encoder": "8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516",
}


class ContractError(ValueError):
    pass


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def verify_contract(
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    evidence_schema: Mapping[str, Any],
    entrypoint_text: str,
) -> dict[str, Any]:
    _require(contract.get("contract_id") == "lam-arc-contextual-successor-v1-execution-lock", "contract id drift")
    _require(contract.get("contract_version") == 1, "contract version drift")
    _require(contract.get("status") == "PREOUTCOME_EXECUTION_LOCK_ONLY", "contract status drift")
    _require(contract.get("protocol_id") == "lam-arc-contextual-successor-v1-draft", "protocol id drift")
    _require(isinstance(contract.get("stack_base_sha"), str) and HEX40.fullmatch(contract["stack_base_sha"]) is not None, "stack base SHA must be lowercase 40-hex")
    _require(contract.get("scientific_execution_authorized") is False, "contract must not authorize scientific execution")
    _require(contract.get("outcome_access_authorized") is False, "contract must not authorize outcome access")
    _require(contract.get("old_locked_test_access") == "PROHIBITED", "old locked ARC-v5 test must remain prohibited")

    _require(protocol.get("protocol_id") == contract.get("protocol_id"), "protocol identity mismatch")
    _require(protocol.get("status") == "DRAFT_NOT_FROZEN", "repository successor protocol must remain draft in this pre-outcome lane")
    _require(protocol.get("execution_authorized") is False, "repository successor protocol must remain unauthorized")
    _require(protocol.get("old_locked_test_access") == "PROHIBITED", "protocol old locked test policy drift")
    _require(protocol.get("proposed_seeds") == EXPECTED_SEEDS, "frozen seed list drift")
    systems = protocol.get("systems")
    _require(isinstance(systems, Mapping), "protocol.systems must be an object")
    _require(list(systems.keys())[:4] == EXPECTED_SYSTEMS, "primary system ordering/identity drift")
    hard_blockers = protocol.get("hard_blockers")
    _require(isinstance(hard_blockers, list), "protocol hard_blockers must be a list")
    _require(EXPECTED_CURRENT_BLOCKERS.issubset(set(hard_blockers)), "a current hard blocker was removed without separate resolution")

    _require(contract.get("frozen_primary_systems") == EXPECTED_SYSTEMS, "contract system set drift")
    _require(contract.get("frozen_seeds") == EXPECTED_SEEDS, "contract seed set drift")
    _require(set(contract.get("current_required_hard_blockers", [])) == EXPECTED_CURRENT_BLOCKERS, "contract blocker set drift")

    dependencies = _mapping(contract.get("frozen_dependencies"), "contract.frozen_dependencies")
    evidence_dep = _mapping(dependencies.get("evidence_schema"), "contract.frozen_dependencies.evidence_schema")
    _require(evidence_dep.get("artifact") == "protocols/arc_successor_v1_evidence_schema.json", "evidence-schema path drift")
    _require(evidence_dep.get("required_status") == "PREOUTCOME_CONTRACT_ONLY", "evidence-schema required status drift")
    _require(evidence_schema.get("status") == "PREOUTCOME_CONTRACT_ONLY", "evidence schema is not pre-outcome-only")
    _require(evidence_schema.get("contract_does_not_authorize_execution") is True, "evidence schema execution boundary drift")
    _require(evidence_schema.get("contract_does_not_authorize_outcome_access") is True, "evidence schema outcome boundary drift")
    _require(evidence_schema.get("frozen_primary_seeds") == EXPECTED_SEEDS, "evidence-schema seed drift")
    _require(evidence_schema.get("allowed_primary_systems") == EXPECTED_SYSTEMS, "evidence-schema system drift")
    for key, expected_hash in EXPECTED_DEPENDENCY_HASHES.items():
        binding = _mapping(dependencies.get(key), f"contract.frozen_dependencies.{key}")
        _require(binding.get("sha256") == expected_hash, f"{key} dependency hash drift")
        _require(HEX64.fullmatch(expected_hash) is not None, f"{key} expected hash malformed")

    _require(set(contract.get("required_source_binding_fields", [])) == EXPECTED_SOURCE_FIELDS, "source-binding field set drift")
    _require(set(contract.get("required_environment_binding_fields", [])) == EXPECTED_ENVIRONMENT_FIELDS, "environment-binding field set drift")
    _require(set(contract.get("required_execution_manifest_fields", [])) == EXPECTED_MANIFEST_FIELDS, "execution-manifest field set drift")

    auth = _mapping(contract.get("required_authorization_state"), "contract.required_authorization_state")
    _require(auth.get("protocol_status") == "FROZEN", "future authorization must require frozen protocol")
    _require(auth.get("protocol_execution_authorized") is True, "future authorization must require explicit protocol execution authorization")
    _require(auth.get("protocol_hard_blockers") == [], "future authorization must require zero hard blockers")
    _require(auth.get("confirmatory_dataset_approved") is True, "future authorization must require confirmatory dataset approval")
    _require(auth.get("independent_freshness_review_complete") is True, "future authorization must require independent freshness review")
    _require(auth.get("outcomes_observed_before_authorization") is False, "outcomes must remain unseen before authorization")

    entrypoint = _mapping(contract.get("preoutcome_entrypoint"), "contract.preoutcome_entrypoint")
    _require(entrypoint.get("artifact") == "tools/arc_successor_locked_entrypoint.py", "entrypoint path drift")
    _require(entrypoint.get("default_mode") == "verify-preoutcome", "default mode must remain pre-outcome verification")
    _require(entrypoint.get("allowed_current_modes") == ["verify-preoutcome"], "current allowed modes must exclude scientific execution")
    _require(entrypoint.get("scientific_runner_bound") is False, "scientific runner must remain unbound")
    _require(entrypoint.get("scientific_runner_identity") is None, "scientific runner identity must remain absent")
    command = entrypoint.get("canonical_preoutcome_command")
    _require(command == "python tools/arc_successor_locked_entrypoint.py --mode verify-preoutcome", "canonical pre-outcome command drift")

    rules = _mapping(contract.get("fail_closed_rules"), "contract.fail_closed_rules")
    for key in (
        "execute_mode_must_refuse_while_scientific_runner_unbound",
        "execute_mode_must_refuse_while_protocol_not_frozen",
        "execute_mode_must_refuse_without_independent_authorization_receipt",
        "execute_mode_must_refuse_without_exact_dataset_bytes",
        "execute_mode_must_refuse_without_exact_environment_lock",
        "execute_mode_must_refuse_without_exact_source_binding",
        "old_arc_v5_locked_test_never_permitted",
        "failed_seeds_must_be_retained",
        "secondary_metric_rescue_prohibited",
        "posthoc_seed_exclusion_prohibited",
    ):
        _require(rules.get(key) is True, f"fail-closed rule weakened: {key}")

    publication = _mapping(contract.get("publication_boundary"), "contract.publication_boundary")
    prohibited = publication.get("prohibited_claims")
    _require(isinstance(prohibited, list) and "execution_authorized" in prohibited, "publication boundary must prohibit authorization claim")

    forbidden_imports = ("import torch", "from torch", "import transformers", "from transformers", "import datasets", "from datasets")
    lower_entrypoint = entrypoint_text.lower()
    for token in forbidden_imports:
        _require(token not in lower_entrypoint, f"pre-outcome entrypoint must remain outcome/model blind: {token}")

    return {
        "status": "ARC_SUCCESSOR_PREOUTCOME_EXECUTION_LOCK_VERIFIED",
        "scientific_execution_authorized": False,
        "outcome_access_authorized": False,
        "old_locked_test_access": "PROHIBITED",
        "protocol_status": protocol.get("status"),
        "hard_blockers_retained": sorted(EXPECTED_CURRENT_BLOCKERS),
        "frozen_systems": EXPECTED_SYSTEMS,
        "frozen_seeds": EXPECTED_SEEDS,
    }


def verify_repository() -> dict[str, Any]:
    contract = _mapping(_load(CONTRACT), "contract")
    protocol = _mapping(_load(PROTOCOL), "protocol")
    evidence_schema = _mapping(_load(EVIDENCE_SCHEMA), "evidence_schema")
    result = verify_contract(contract, protocol, evidence_schema, ENTRYPOINT.read_text(encoding="utf-8"))
    result["execution_contract_sha256"] = _sha256(CONTRACT)
    result["evidence_schema_sha256"] = _sha256(EVIDENCE_SCHEMA)
    result["entrypoint_sha256"] = _sha256(ENTRYPOINT)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = verify_repository()
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "FAIL_PREOUTCOME_EXECUTION_LOCK", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
