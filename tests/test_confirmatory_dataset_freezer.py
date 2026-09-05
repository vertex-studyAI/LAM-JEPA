from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirmatory_dataset_freezer import freeze_receipt, validate_manifest  # noqa: E402


def h(ch: str) -> str:
    return ch * 64


def valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "status": "FROZEN_PREOUTCOME",
        "execution_authorized": False,
        "outcomes_observed": False,
        "dataset": {
            "name": "Fresh Science Benchmark",
            "source": "https://example.org/fresh-science",
            "revision": "revision-2026-09-01",
            "license": "CC-BY-4.0",
            "task_type": "four_choice_science_multiple_choice",
            "answer_choice_count": 4,
            "content_sha256": h("a"),
            "provenance_sha256": h("b"),
        },
        "adapter": {
            "path": "tools/fresh_science_adapter.py",
            "sha256": h("c"),
            "label_blind": True,
            "deterministic": True,
        },
        "selection": {
            "item_count": 300,
            "selected_item_ids_sha256": h("d"),
            "selection_rationale_sha256": h("e"),
            "labels_hidden_from_development": True,
            "selection_completed_before_treatment_outcomes": True,
        },
        "freshness": {
            "project_treatment_family_tuned_on_dataset": False,
            "project_outcomes_previously_observed": False,
            "prior_access_audit_completed": True,
            "prior_access_audit_sha256": h("f"),
        },
        "overlap": {
            "development_overlap_count": 0,
            "historical_arc_overlap_count": 0,
            "audit_sha256": h("1"),
            "semantic_overlap_reviewed": True,
        },
        "one_shot_policy": {
            "maximum_confirmatory_runs": 1,
            "hyperparameter_updates_after_access": False,
            "architecture_changes_after_access": False,
            "retain_all_outputs": True,
            "retain_negative_or_inconclusive_result": True,
        },
        "independent_review": {
            "approved": True,
            "reviewer": "reviewer-1",
            "reviewed_at": "2026-09-06T00:00:00Z",
            "artifact_sha256": h("2"),
            "outcomes_available_to_reviewer": False,
        },
    }


class ConfirmatoryDatasetFreezerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = valid_manifest()

    def assert_rejected(self, manifest: dict, needle: str) -> None:
        errors = validate_manifest(manifest)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_valid_manifest_freezes_deterministically_without_authorizing_execution(self) -> None:
        self.assertEqual([], validate_manifest(self.manifest))
        first = freeze_receipt(self.manifest)
        second = freeze_receipt(copy.deepcopy(self.manifest))
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))
        self.assertFalse(self.manifest["execution_authorized"])

    def test_historical_arc_name_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["dataset"]["name"] = "AI2 ARC-Challenge"
        self.assert_rejected(manifest, "historical AI2 ARC-Challenge")

    def test_historical_arc_split_hash_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["dataset"]["content_sha256"] = "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05"
        self.assert_rejected(manifest, "historical ARC-Challenge split")

    def test_outcome_observation_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["outcomes_observed"] = True
        self.assert_rejected(manifest, "outcomes_observed")

    def test_development_tuning_on_candidate_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["freshness"]["project_treatment_family_tuned_on_dataset"] = True
        self.assert_rejected(manifest, "project_treatment_family_tuned_on_dataset")

    def test_overlap_with_development_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["overlap"]["development_overlap_count"] = 1
        self.assert_rejected(manifest, "development_overlap_count")

    def test_hidden_labels_are_required(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["selection"]["labels_hidden_from_development"] = False
        self.assert_rejected(manifest, "labels_hidden_from_development")

    def test_one_shot_policy_cannot_be_relaxed(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["one_shot_policy"]["maximum_confirmatory_runs"] = 2
        self.assert_rejected(manifest, "maximum_confirmatory_runs")

    def test_independent_review_is_required_and_preoutcome(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["independent_review"]["approved"] = False
        manifest["independent_review"]["outcomes_available_to_reviewer"] = True
        errors = validate_manifest(manifest)
        self.assertTrue(any("approved" in error for error in errors), errors)
        self.assertTrue(any("outcomes_available_to_reviewer" in error for error in errors), errors)

    def test_task_must_remain_four_choice_science(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["dataset"]["task_type"] = "generic_multiple_choice"
        manifest["dataset"]["answer_choice_count"] = 5
        errors = validate_manifest(manifest)
        self.assertTrue(any("task_type" in error for error in errors), errors)
        self.assertTrue(any("answer_choice_count" in error for error in errors), errors)

    def test_placeholder_identity_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["dataset"]["revision"] = "TBD"
        self.assert_rejected(manifest, "dataset.revision")


if __name__ == "__main__":
    unittest.main()
