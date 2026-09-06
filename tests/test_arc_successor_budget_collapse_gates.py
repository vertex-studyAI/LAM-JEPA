#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_arc_successor_budget_collapse_gates import GateError, verify_gates  # noqa: E402

GATES = json.loads((ROOT / "protocols" / "arc_successor_v1_budget_collapse_gates.json").read_text(encoding="utf-8"))
PROTOCOL = json.loads((ROOT / "protocols" / "arc_successor_v1_draft.json").read_text(encoding="utf-8"))


class BudgetCollapseGateTests(unittest.TestCase):
    def verify(self, gates=None, protocol=None):
        return verify_gates(copy.deepcopy(gates or GATES), copy.deepcopy(protocol or PROTOCOL))

    def assert_rejected(self, mutate_gates=None, mutate_protocol=None):
        gates = copy.deepcopy(GATES)
        protocol = copy.deepcopy(PROTOCOL)
        if mutate_gates:
            mutate_gates(gates)
        if mutate_protocol:
            mutate_protocol(protocol)
        with self.assertRaises(GateError):
            verify_gates(gates, protocol)

    def test_canonical_contract_passes_without_authorization(self):
        result = self.verify()
        self.assertEqual(result["status"], "ARC_SUCCESSOR_BUDGET_COLLAPSE_GATES_VERIFIED_PREOUTCOME")
        self.assertFalse(result["scientific_execution_authorized"])
        self.assertFalse(result["outcome_access_authorized"])
        self.assertTrue(result["canonical_parent_blocker_ledger_unchanged"])

    def test_execution_authorization_drift_fails(self):
        self.assert_rejected(lambda g: g.__setitem__("scientific_execution_authorized", True))

    def test_heldout_probe_access_fails(self):
        self.assert_rejected(lambda g: g["measurement_contract"].__setitem__("heldout_confirmatory_examples_permitted", True))

    def test_collapse_threshold_drift_fails(self):
        self.assert_rejected(lambda g: g["collapse_thresholds"]["normalized_effective_rank"].__setitem__("minimum_inclusive", 0.01))
        self.assert_rejected(lambda g: g["collapse_thresholds"]["active_latent_dimension_fraction"].__setitem__("minimum_fraction_inclusive", 0.01))
        self.assert_rejected(lambda g: g["collapse_thresholds"]["mean_absolute_pairwise_cosine_similarity"].__setitem__("maximum_exclusive", 1.0))

    def test_seed_scope_drift_fails(self):
        self.assert_rejected(lambda g: g["collapse_thresholds"].__setitem__("seed_scope", [11, 23, 37, 53]))

    def test_parameter_budget_weakening_fails(self):
        self.assert_rejected(lambda g: g["parameter_match_tolerance"].__setitem__("maximum_t1_to_b0_gradient_active_parameter_ratio_inclusive", 1.25))
        self.assert_rejected(lambda g: g["parameter_match_tolerance"].__setitem__("must_include_t1_predictor", False))

    def test_compute_budget_weakening_fails(self):
        self.assert_rejected(lambda g: g["compute_budget"].__setitem__("per_seed_maximum_t1_to_b0_accelerator_seconds_ratio_inclusive", 3.0))
        self.assert_rejected(lambda g: g["compute_budget"].__setitem__("optimizer_steps_ratio", 1.1))

    def test_rescue_and_posthoc_changes_fail(self):
        self.assert_rejected(lambda g: g["decision_integration"].__setitem__("secondary_metric_rescue", True))
        self.assert_rejected(lambda g: g["decision_integration"].__setitem__("secondary_system_rescue", True))
        self.assert_rejected(lambda g: g["decision_integration"].__setitem__("posthoc_seed_exclusion", True))
        self.assert_rejected(lambda g: g["decision_integration"].__setitem__("posthoc_threshold_movement", True))

    def test_parent_blocker_removal_is_rejected_in_stacked_lane(self):
        def mutate(protocol):
            protocol["hard_blockers"].remove("MAX_COMPUTE_RATIO")
        self.assert_rejected(mutate_protocol=mutate)

    def test_old_locked_test_reopening_fails(self):
        self.assert_rejected(lambda g: g.__setitem__("old_locked_test_access", "ALLOWED"))

    def test_nonfinite_numeric_gate_fails(self):
        self.assert_rejected(lambda g: g["compute_budget"].__setitem__("per_seed_maximum_t1_to_b0_accelerator_seconds_ratio_inclusive", float("nan")))


if __name__ == "__main__":
    unittest.main()
