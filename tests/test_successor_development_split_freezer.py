from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from successor_development_split_freezer import (  # noqa: E402
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    canonical_sha256,
    freeze_partition,
    main,
    validate_config,
    validate_eligible_ids,
)


def h(ch: str) -> str:
    return ch * 64


def ids() -> list[str]:
    return [f"ARC_TRAIN_{i:04d}" for i in range(EXPECTED_ELIGIBLE_ROWS)]


def valid_config() -> dict:
    return {
        "schema_version": 1,
        "status": "FROZEN_PREOUTCOME_DEVELOPMENT_ONLY",
        "execution_authorized": False,
        "outcomes_observed": False,
        "source": {
            "dataset": "AI2 ARC-Challenge",
            "split": "train",
            "source_sha256": EXPECTED_SOURCE_SHA256,
            "source_rows": EXPECTED_SOURCE_ROWS,
            "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
            "eligibility_rule": "exactly_four_answer_choices",
            "eligibility_artifact_sha256": h("a"),
        },
        "construction": {
            "method": "sha256_rank_v1",
            "seed": "successor-dev-v1-preoutcome",
            "dev_count": 224,
            "label_blind": True,
            "preserve_source_order_in_outputs": True,
        },
        "boundary": {
            "development_only": True,
            "arc_validation_used": False,
            "arc_test_used": False,
            "same_partition_for_B0_B1_T1_T2": True,
            "partition_selection_informed_by_treatment_outcomes": False,
            "metric_selection_policy_sha256": h("b"),
        },
    }


class DevelopmentSplitFreezerTests(unittest.TestCase):
    def test_valid_config_and_ids_freeze_deterministically(self) -> None:
        cfg = valid_config()
        eligible = ids()
        self.assertEqual(validate_config(cfg), [])
        self.assertEqual(validate_eligible_ids(eligible), [])
        first = freeze_partition(cfg, eligible)
        second = freeze_partition(copy.deepcopy(cfg), list(eligible))
        self.assertEqual(first, second)
        self.assertEqual(first["partition"]["train_count"], EXPECTED_ELIGIBLE_ROWS - 224)
        self.assertEqual(first["partition"]["dev_count"], 224)
        self.assertEqual(first["partition"]["overlap_count"], 0)
        self.assertEqual(first["partition"]["coverage_count"], EXPECTED_ELIGIBLE_ROWS)
        self.assertEqual(
            first["partition"]["eligible_ids_sha256"],
            canonical_sha256(eligible),
        )
        self.assertFalse(first["execution_authorized"])
        self.assertTrue(first["boundary"]["development_only"])

    def test_output_lists_preserve_original_order(self) -> None:
        eligible = ids()
        result = freeze_partition(valid_config(), eligible)
        positions = {item_id: i for i, item_id in enumerate(eligible)}
        self.assertEqual(
            result["partition"]["train_ids"],
            sorted(result["partition"]["train_ids"], key=positions.__getitem__),
        )
        self.assertEqual(
            result["partition"]["dev_ids"],
            sorted(result["partition"]["dev_ids"], key=positions.__getitem__),
        )

    def test_rejects_validation_or_test_source(self) -> None:
        for split in ("validation", "test"):
            cfg = valid_config()
            cfg["source"]["split"] = split
            self.assertTrue(any("source.split" in e for e in validate_config(cfg)))

    def test_rejects_arc_source_hash_drift(self) -> None:
        cfg = valid_config()
        cfg["source"]["source_sha256"] = h("c")
        self.assertTrue(any("source.source_sha256" in e for e in validate_config(cfg)))

    def test_rejects_outcome_informed_partition(self) -> None:
        cfg = valid_config()
        cfg["boundary"]["partition_selection_informed_by_treatment_outcomes"] = True
        self.assertTrue(any("partition_selection_informed" in e for e in validate_config(cfg)))

    def test_rejects_placeholder_seed(self) -> None:
        cfg = valid_config()
        cfg["construction"]["seed"] = "TBD"
        self.assertTrue(any("construction.seed" in e for e in validate_config(cfg)))

    def test_rejects_duplicate_ids(self) -> None:
        eligible = ids()
        eligible[-1] = eligible[0]
        self.assertTrue(any("duplicates" in e for e in validate_eligible_ids(eligible)))

    def test_rejects_wrong_eligible_count(self) -> None:
        self.assertTrue(any("exactly" in e for e in validate_eligible_ids(ids()[:-1])))

    def test_seed_change_changes_partition(self) -> None:
        eligible = ids()
        a = freeze_partition(valid_config(), eligible)
        cfg = valid_config()
        cfg["construction"]["seed"] = "different-preoutcome-seed"
        b = freeze_partition(cfg, eligible)
        self.assertNotEqual(a["partition"]["dev_ids_sha256"], b["partition"]["dev_ids_sha256"])

    def test_cli_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = td / "config.json"
            eligible = td / "ids.json"
            out = td / "split.json"
            cfg.write_text(json.dumps(valid_config()), encoding="utf-8")
            eligible.write_text(json.dumps(ids()), encoding="utf-8")
            self.assertEqual(main(["--config", str(cfg), "--eligible-ids", str(eligible), "--output", str(out)]), 0)
            with self.assertRaises(FileExistsError):
                main(["--config", str(cfg), "--eligible-ids", str(eligible), "--output", str(out)])


if __name__ == "__main__":
    unittest.main()
