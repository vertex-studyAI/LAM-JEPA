#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arc_successor_locked_entrypoint import execution_refusal_reasons  # noqa: E402
from verify_arc_successor_execution_contract import (  # noqa: E402
    CONTRACT,
    EVIDENCE_SCHEMA,
    ENTRYPOINT,
    PROTOCOL,
    ContractError,
    verify_contract,
    verify_repository,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class SuccessorExecutionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load(CONTRACT)
        self.protocol = load(PROTOCOL)
        self.evidence = load(EVIDENCE_SCHEMA)
        self.entrypoint_text = ENTRYPOINT.read_text(encoding="utf-8")

    def assert_rejected(self, contract=None, protocol=None, evidence=None, entrypoint_text=None) -> None:
        with self.assertRaises(ContractError):
            verify_contract(
                contract if contract is not None else self.contract,
                protocol if protocol is not None else self.protocol,
                evidence if evidence is not None else self.evidence,
                entrypoint_text if entrypoint_text is not None else self.entrypoint_text,
            )

    def test_repository_contract_is_valid_and_preoutcome_only(self) -> None:
        result = verify_repository()
        self.assertEqual(result["status"], "ARC_SUCCESSOR_PREOUTCOME_EXECUTION_LOCK_VERIFIED")
        self.assertFalse(result["scientific_execution_authorized"])
        self.assertFalse(result["outcome_access_authorized"])
        self.assertEqual(result["old_locked_test_access"], "PROHIBITED")

    def test_contract_cannot_preclaim_execution_or_outcome_access(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["scientific_execution_authorized"] = True
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["outcome_access_authorized"] = True
        self.assert_rejected(contract=mutated)

    def test_current_protocol_must_remain_draft_and_unauthorized(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["status"] = "FROZEN"
        self.assert_rejected(protocol=mutated)
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_authorized"] = True
        self.assert_rejected(protocol=mutated)

    def test_current_hard_blockers_cannot_be_silently_dropped(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["hard_blockers"].remove("EXACT_REPRODUCE_COMMAND")
        self.assert_rejected(protocol=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["current_required_hard_blockers"].remove("CONFIRMATORY_DATASET")
        self.assert_rejected(contract=mutated)

    def test_seed_and_system_freezes_cannot_drift(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["frozen_seeds"] = [11, 23, 37, 53, 99]
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["frozen_primary_systems"] = ["B0", "B1", "T1"]
        self.assert_rejected(contract=mutated)

    def test_source_environment_and_manifest_contracts_fail_closed(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["required_source_binding_fields"].remove("source_tree_sha")
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["required_environment_binding_fields"].remove("host_metadata_sha256")
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["required_execution_manifest_fields"].remove("authorization_receipt_sha256")
        self.assert_rejected(contract=mutated)

    def test_old_arc_v5_locked_test_cannot_be_reopened(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["old_locked_test_access"] = "ALLOWED"
        self.assert_rejected(protocol=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["old_locked_test_access"] = "ALLOWED"
        self.assert_rejected(contract=mutated)

    def test_evidence_schema_must_remain_preoutcome_and_nonauthorizing(self) -> None:
        mutated = copy.deepcopy(self.evidence)
        mutated["status"] = "COMPLETE_RESULT_PACKAGE"
        self.assert_rejected(evidence=mutated)
        mutated = copy.deepcopy(self.evidence)
        mutated["contract_does_not_authorize_outcome_access"] = False
        self.assert_rejected(evidence=mutated)

    def test_entrypoint_cannot_bind_a_scientific_runner_in_this_lane(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["preoutcome_entrypoint"]["scientific_runner_bound"] = True
        mutated["preoutcome_entrypoint"]["scientific_runner_identity"] = "tools/run_science.py"
        self.assert_rejected(contract=mutated)

    def test_preoutcome_entrypoint_stays_model_and_dataset_blind(self) -> None:
        self.assert_rejected(entrypoint_text=self.entrypoint_text + "\nimport torch\n")
        self.assert_rejected(entrypoint_text=self.entrypoint_text + "\nfrom transformers import AutoModel\n")
        self.assert_rejected(entrypoint_text=self.entrypoint_text + "\nimport datasets\n")

    def test_execute_mode_is_refused_by_current_repository_state(self) -> None:
        reasons = execution_refusal_reasons(self.protocol, self.contract)
        self.assertIn("protocol_not_frozen", reasons)
        self.assertIn("protocol_execution_not_authorized", reasons)
        self.assertIn("protocol_hard_blockers_present", reasons)
        self.assertIn("scientific_runner_unbound", reasons)
        self.assertIn("execution_contract_not_authorized", reasons)
        self.assertIn("outcome_access_not_authorized", reasons)
        self.assertIn("authorization_receipt_missing", reasons)
        self.assertIn("execution_manifest_missing", reasons)

    def test_future_authorization_contract_requires_all_independent_gates(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["required_authorization_state"]["confirmatory_dataset_approved"] = False
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["required_authorization_state"]["independent_freshness_review_complete"] = False
        self.assert_rejected(contract=mutated)
        mutated = copy.deepcopy(self.contract)
        mutated["required_authorization_state"]["protocol_hard_blockers"] = ["EXACT_REPRODUCE_COMMAND"]
        self.assert_rejected(contract=mutated)


if __name__ == "__main__":
    unittest.main()
