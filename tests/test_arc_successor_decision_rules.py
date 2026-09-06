from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_arc_successor_decision_rules import DecisionRuleError, verify_rules  # noqa: E402

RULES_PATH = ROOT / "protocols" / "arc_successor_v1_decision_rules.json"
SUCCESSOR_PATH = ROOT / "protocols" / "arc_successor_v1_draft.json"
EXPECTED_RULES_SHA256 = "28d7158589091c2326154a30a4a14b1dac50c5d3049fae2b70298ac1a14dc625"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


class ArcSuccessorDecisionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = _load(RULES_PATH)
        self.successor = _load(SUCCESSOR_PATH)

    def assert_rejected(self, rules: dict | None = None, successor: dict | None = None, needle: str = "") -> None:
        with self.assertRaises(DecisionRuleError) as caught:
            verify_rules(rules or self.rules, successor or self.successor)
        self.assertIn(needle, str(caught.exception))

    def test_repository_rules_verify_and_are_exactly_sha_bound(self) -> None:
        result = verify_rules(self.rules, self.successor)
        self.assertEqual(result["status"], "ARC_SUCCESSOR_PREOUTCOME_DECISION_RULES_VERIFIED")
        self.assertFalse(result["execution_authorized"])
        actual = hashlib.sha256(RULES_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, EXPECTED_RULES_SHA256)
        for blocker in ("DELTA_PRIMARY", "SEED_WIN_FRACTION", "UNCERTAINTY_RULE"):
            self.assertEqual(self.successor["resolved_blockers"][blocker]["sha256"], actual)

    def test_more_permissive_material_effect_is_rejected(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["material_effect"]["delta_primary_absolute_accuracy"] = 0.01
        self.assert_rejected(rules=rules, needle="DELTA_PRIMARY")

    def test_three_of_five_seed_gate_is_rejected(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["seed_consistency"]["minimum_positive_seed_count"] = 3
        self.assert_rejected(rules=rules, needle="minimum positive seed count")

    def test_seed_ties_cannot_be_promoted_to_wins(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["seed_consistency"]["ties_are_wins"] = True
        self.assert_rejected(rules=rules, needle="ties must not count as wins")

    def test_bootstrap_replicate_or_seed_drift_is_rejected(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["uncertainty"]["replicates"] = 1000
        self.assert_rejected(rules=rules, needle="uncertainty.replicates")
        rules = copy.deepcopy(self.rules)
        rules["uncertainty"]["bootstrap_seed"] = 7
        self.assert_rejected(rules=rules, needle="uncertainty.bootstrap_seed")

    def test_unpaired_uncertainty_is_rejected(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["uncertainty"]["pairing_preserved"] = False
        self.assert_rejected(rules=rules, needle="uncertainty.pairing_preserved")

    def test_secondary_metric_rescue_is_rejected(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["decision_logic"]["secondary_metrics_can_rescue_primary_failure"] = True
        self.assert_rejected(rules=rules, needle="secondary rescue")

    def test_decision_rules_cannot_authorize_execution(self) -> None:
        rules = copy.deepcopy(self.rules)
        rules["execution_authorized"] = True
        self.assert_rejected(rules=rules, needle="must not authorize execution")

    def test_successor_must_keep_all_nondecision_blockers(self) -> None:
        successor = copy.deepcopy(self.successor)
        successor["hard_blockers"].remove("CONFIRMATORY_DATASET")
        self.assert_rejected(successor=successor, needle="hard_blockers")

    def test_resolved_blocker_binding_cannot_point_elsewhere(self) -> None:
        successor = copy.deepcopy(self.successor)
        successor["resolved_blockers"]["DELTA_PRIMARY"]["artifact"] = "results.json"
        self.assert_rejected(successor=successor, needle="artifact binding drift")


if __name__ == "__main__":
    unittest.main()
