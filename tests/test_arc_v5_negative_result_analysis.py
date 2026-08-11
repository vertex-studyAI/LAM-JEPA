from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analysis" / "analyze_arc_v5_negative_result.py"
spec = importlib.util.spec_from_file_location("arc_negative", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def row(item_id: str, label: int, prediction: int) -> dict:
    return {"id": item_id, "label": label, "prediction": prediction}


def package() -> dict:
    ids = ["a", "b", "c", "d"]
    labels = [0, 1, 0, 1]
    predictions = {
        "legacy_ce": {
            1: [0, 0, 0, 1],
            2: [1, 1, 0, 1],
        },
        "repaired_v5_ce": {
            1: [0, 1, 1, 1],
            2: [0, 1, 0, 0],
        },
        "no_quantizer_ce": {
            1: [0, 1, 0, 1],
            2: [0, 0, 0, 1],
        },
        "repaired_v5_shuffled_labels": {
            1: [1, 0, 1, 0],
            2: [1, 0, 1, 0],
        },
    }
    records = {}
    for condition, by_seed in predictions.items():
        records[condition] = []
        for seed, preds in by_seed.items():
            rows = [row(item_id, label, prediction) for item_id, label, prediction in zip(ids, labels, preds, strict=True)]
            records[condition].append({"seed": seed, "rows": rows})
    return {
        "claim_boundary": {"test_accessed": False, "research_complete": False},
        "records": records,
        "verdict": "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION",
    }


class NegativeSliceAnalysisTests(unittest.TestCase):
    def test_pairwise_fixed_and_regressed_are_counted(self) -> None:
        analysis = module.analyze_result(package())
        overall = analysis["repaired_vs_legacy"]["overall"]
        self.assertEqual(overall["fixed"], 2)
        self.assertEqual(overall["regressed"], 2)
        self.assertEqual(sum(overall.values()), 8)

    def test_per_label_and_stability_slices_are_retained(self) -> None:
        analysis = module.analyze_result(package())
        repaired = analysis["conditions"]["repaired_v5_ce"]
        self.assertEqual(repaired["per_true_label"]["0"]["count"], 4)
        self.assertEqual(repaired["per_true_label"]["1"]["count"], 4)
        stability = analysis["repaired_item_stability"]
        self.assertEqual(sum(stability["correct_seed_count_histogram"].values()), 4)
        self.assertEqual(len(stability["items_hardest_first"]), 4)

    def test_misaligned_ids_fail_closed(self) -> None:
        result = package()
        result["records"]["no_quantizer_ce"][0]["rows"][0]["id"] = "wrong-id"
        with self.assertRaisesRegex(ValueError, "row identity/order mismatch"):
            module.analyze_result(result)

    def test_test_access_or_research_complete_fail_closed(self) -> None:
        result = package()
        result["claim_boundary"]["test_accessed"] = True
        with self.assertRaisesRegex(ValueError, "test_accessed=false"):
            module.analyze_result(result)
        result = package()
        result["claim_boundary"]["research_complete"] = True
        with self.assertRaisesRegex(ValueError, "research_complete=true"):
            module.analyze_result(result)


if __name__ == "__main__":
    unittest.main()
