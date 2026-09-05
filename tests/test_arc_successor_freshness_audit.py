from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

from verify_arc_successor_freshness_audit import validate_audit  # noqa: E402

AUDIT_PATH = ROOT / "protocols" / "data_freshness_audit_v1.json"


class FreshnessAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def assert_rejected(self, audit: dict, needle: str) -> None:
        errors = validate_audit(ROOT, audit)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_repository_audit_validates(self) -> None:
        self.assertEqual([], validate_audit(ROOT, self.audit))

    def test_cannot_mark_validation_fresh(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["arc_challenge"]["splits"]["validation"]["confirmatory_fresh"] = True
        self.assert_rejected(audit, "validation.confirmatory_fresh")

    def test_cannot_authorize_old_arc_test(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["successor_confirmatory_surface"]["old_arc_test_allowed"] = True
        self.assert_rejected(audit, "old ARC test")

    def test_freshness_audit_cannot_authorize_execution(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["execution_authorized"] = True
        self.assert_rejected(audit, "must not authorize execution")

    def test_confirmatory_dataset_cannot_be_silently_filled(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["successor_confirmatory_surface"]["dataset"] = "candidate-dataset"
        self.assert_rejected(audit, "must remain null until separately frozen")

    def test_manifest_hash_drift_is_rejected(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["arc_challenge"]["splits"]["train"]["sha256"] = "0" * 64
        self.assert_rejected(audit, "does not match frozen ARC manifest")

    def test_conservative_whole_split_rule_cannot_be_disabled(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["whole_split_conservative_rule"] = False
        self.assert_rejected(audit, "whole_split_conservative_rule")

    def test_claim_boundary_must_stay_pending_review(self) -> None:
        audit = copy.deepcopy(self.audit)
        audit["blocker_disposition"]["DATA_FRESHNESS_AUDIT"] = "RESOLVED"
        self.assert_rejected(audit, "blocker_disposition drift")


if __name__ == "__main__":
    unittest.main()
