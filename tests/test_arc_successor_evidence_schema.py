from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_arc_successor_evidence_schema import ContractError, validate  # noqa: E402

CONTRACT_PATH = ROOT / "protocols" / "arc_successor_v1_evidence_schema.json"
TEMPLATE_PATH = ROOT / "protocols" / "arc_successor_v1_run_receipt_template.json"


class ArcSuccessorEvidenceSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

    def test_repository_contract_verifies_preoutcome_only(self) -> None:
        observed = validate(self.contract, self.template, ROOT)
        self.assertEqual(
            observed["decision_rules"],
            "28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625",
        )
        self.assertFalse(self.template["authorization"]["execution_authorized"])
        self.assertFalse(self.template["authorization"]["outcome_access_authorized"])
        self.assertEqual(self.template["artifacts"], [])
        self.assertEqual(self.template["claim_ledger"], [])

    def test_seed_or_system_drift_fails_closed(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["frozen_primary_seeds"] = [11, 23, 37, 53]
        with self.assertRaisesRegex(ContractError, "seed set/order drift"):
            validate(contract, self.template, ROOT)

        contract = copy.deepcopy(self.contract)
        contract["allowed_primary_systems"] = ["B0", "T1"]
        with self.assertRaisesRegex(ContractError, "system set/order drift"):
            validate(contract, self.template, ROOT)

    def test_authorization_cannot_be_preclaimed(self) -> None:
        template = copy.deepcopy(self.template)
        template["authorization"]["execution_authorized"] = True
        with self.assertRaisesRegex(ContractError, "authorization must remain false"):
            validate(self.contract, template, ROOT)

        template = copy.deepcopy(self.template)
        template["authorization"]["outcome_access_authorized"] = True
        with self.assertRaisesRegex(ContractError, "authorization must remain false"):
            validate(self.contract, template, ROOT)

    def test_result_artifacts_or_claims_cannot_appear_in_preoutcome_template(self) -> None:
        template = copy.deepcopy(self.template)
        template["artifacts"] = [{"role": "metric_json"}]
        with self.assertRaisesRegex(ContractError, "must not contain result artifacts"):
            validate(self.contract, template, ROOT)

        template = copy.deepcopy(self.template)
        template["claim_ledger"] = [{"claim_id": "premature"}]
        with self.assertRaisesRegex(ContractError, "must not contain scientific claims"):
            validate(self.contract, template, ROOT)

    def test_confirmatory_dataset_identity_cannot_be_synthesized(self) -> None:
        template = copy.deepcopy(self.template)
        template["data"]["dataset_id"] = "unreviewed-candidate"
        with self.assertRaisesRegex(ContractError, "confirmatory dataset must remain unresolved"):
            validate(self.contract, template, ROOT)

        template = copy.deepcopy(self.template)
        template["data"]["dataset_bytes_sha256"] = "0" * 64
        with self.assertRaisesRegex(ContractError, "dataset bytes must remain unresolved"):
            validate(self.contract, template, ROOT)

    def test_negative_result_and_no_rescue_guards_are_mandatory(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["claim_ledger_contract"]["secondary_metric_rescue_prohibited"] = False
        with self.assertRaisesRegex(ContractError, "claim-ledger guard missing"):
            validate(contract, self.template, ROOT)

        contract = copy.deepcopy(self.contract)
        contract["integrity_rules"]["posthoc_seed_exclusion"] = True
        with self.assertRaisesRegex(ContractError, "posthoc seed exclusion must remain false"):
            validate(contract, self.template, ROOT)

    def test_old_locked_test_access_remains_prohibited(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["integrity_rules"]["old_locked_test_access"] = "ALLOWED"
        with self.assertRaisesRegex(ContractError, "old locked test must remain prohibited"):
            validate(contract, self.template, ROOT)


if __name__ == "__main__":
    unittest.main()
