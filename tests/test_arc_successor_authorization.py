from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arc_successor_authorization import authorization_receipt, validate_authorization  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hex64(char: str) -> str:
    return char * 64


def frozen_protocol_bytes() -> bytes:
    resolved = {
        key: {"artifact": f"evidence/{key.lower()}.json", "sha256": hex64("a")}
        for key in (
            "DATA_FRESHNESS_AUDIT",
            "CONFIRMATORY_DATASET",
            "ENCODER_FAMILY_AND_REVISION",
            "CONTEXT_TARGET_CONSTRUCTION",
            "DELTA_PRIMARY",
            "SEED_WIN_FRACTION",
            "UNCERTAINTY_RULE",
            "COLLAPSE_THRESHOLDS",
            "PARAMETER_MATCH_TOLERANCE",
            "MAX_COMPUTE_RATIO",
            "EXACT_REPRODUCE_COMMAND",
        )
    }
    protocol = {
        "protocol_id": "lam-arc-contextual-successor-v1",
        "status": "FROZEN",
        "execution_authorized": True,
        "hard_blockers": [],
        "resolved_blockers": resolved,
        "systems": {"B0": "baseline", "B1": "control", "T1": "treatment", "T2": "secondary"},
        "proposed_seeds": [11, 23, 37, 53, 71],
        "integrity_rules": {
            "retain_failed_seeds": True,
            "posthoc_seed_exclusion": False,
            "heldout_threshold_tuning": False,
            "architecture_shopping_after_failure": False,
            "secondary_metric_rescue": False,
            "vq_rescue": False,
        },
    }
    return (json.dumps(protocol, sort_keys=True, indent=2) + "\n").encode()


def valid_manifest(protocol_bytes: bytes) -> dict:
    return {
        "protocol_version": "arc-successor-v1",
        "protocol_sha256": digest(protocol_bytes),
        "dataset_snapshot_sha256": hex64("b"),
        "split_manifest_sha256": hex64("c"),
        "context_target_sha256": hex64("d"),
        "optimizer_contract_sha256": hex64("e"),
        "budget_contract_sha256": hex64("f"),
        "environment_sha256": hex64("1"),
        "analysis_sha256": hex64("2"),
        "exact_reproduce_command": "python -m lam_jepa.arc_successor --config frozen.json",
        "encoder": {
            "family": "encoder-family-x",
            "revision": "revision-1",
            "runtime": "runtime-1",
            "tokenizer_sha256": hex64("3"),
        },
        "systems": {
            "B0": {"implementation": "supervised", "revision": "r1", "runtime": "runtime-1", "config_sha256": hex64("4")},
            "B1": {"implementation": "reconstruction-control", "revision": "r1", "runtime": "runtime-1", "config_sha256": hex64("5")},
            "T1": {"implementation": "distinct-target-jepa", "revision": "r1", "runtime": "runtime-1", "config_sha256": hex64("6")},
            "T2": {"implementation": "jepa-anticollapse", "revision": "r1", "runtime": "runtime-1", "config_sha256": hex64("7")},
        },
        "seeds": [11, 23, 37, 53, 71],
        "matched_budget": {
            "optimizer_steps": 1000,
            "hyperparameter_trials": 8,
            "accelerator_seconds_ceiling": 3600,
        },
        "outcomes_observed": False,
        "independent_review": {
            "approved": True,
            "reviewer": "reviewer-1",
            "reviewed_at": "2026-09-06T00:00:00Z",
            "artifact_sha256": hex64("8"),
        },
    }


class AuthorizationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = frozen_protocol_bytes()
        self.manifest = valid_manifest(self.protocol)

    def assert_rejected(self, manifest: dict, needle: str, protocol: bytes | None = None) -> None:
        errors = validate_authorization(protocol or self.protocol, manifest)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_valid_frozen_inputs_authorize_with_deterministic_receipt(self) -> None:
        self.assertEqual([], validate_authorization(self.protocol, self.manifest))
        first = authorization_receipt(self.protocol, self.manifest)
        second = authorization_receipt(self.protocol, copy.deepcopy(self.manifest))
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_placeholder_encoder_identity_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["encoder"]["family"] = "TBD"
        self.assert_rejected(manifest, "encoder.family")

    def test_protocol_hash_mismatch_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["protocol_sha256"] = hex64("9")
        self.assert_rejected(manifest, "does not match protocol bytes")

    def test_missing_system_binding_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        del manifest["systems"]["B1"]
        self.assert_rejected(manifest, "exactly B0, B1, T1, and T2")

    def test_seed_drift_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"][-1] = 99
        self.assert_rejected(manifest, "exactly match the frozen protocol seed list")

    def test_observed_outcomes_are_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["outcomes_observed"] = True
        self.assert_rejected(manifest, "outcomes_observed")

    def test_missing_independent_review_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["independent_review"]["approved"] = False
        self.assert_rejected(manifest, "independent_review.approved")

    def test_tampered_protocol_is_rejected_by_hash_binding(self) -> None:
        tampered = self.protocol + b"\n"
        self.assert_rejected(self.manifest, "does not match protocol bytes", protocol=tampered)

    def test_receipt_changes_when_frozen_manifest_changes(self) -> None:
        other = copy.deepcopy(self.manifest)
        other["matched_budget"]["accelerator_seconds_ceiling"] = 3700
        self.assertNotEqual(
            authorization_receipt(self.protocol, self.manifest),
            authorization_receipt(self.protocol, other),
        )

    def test_repository_draft_is_deliberately_non_authorizable(self) -> None:
        draft_path = ROOT / "protocols" / "arc_successor_v1_draft.json"
        if not draft_path.exists():
            self.skipTest("repository draft fixture not present")
        draft = draft_path.read_bytes()
        manifest = valid_manifest(draft)
        errors = validate_authorization(draft, manifest)
        self.assertTrue(any("status must equal FROZEN" in error for error in errors), errors)
        self.assertTrue(any("hard_blockers must be empty" in error for error in errors), errors)
        self.assertTrue(any("unresolved draft/freeze markers" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
