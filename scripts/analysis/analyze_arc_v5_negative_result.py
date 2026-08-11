from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

CONDITIONS = (
    "legacy_ce",
    "repaired_v5_ce",
    "no_quantizer_ce",
    "repaired_v5_shuffled_labels",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows_by_seed(result: dict[str, Any], condition: str) -> dict[int, list[dict[str, Any]]]:
    records = result.get("records", {}).get(condition)
    _require(isinstance(records, list) and records, f"missing records for {condition}")
    output: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        seed = int(record["seed"])
        _require(seed not in output, f"duplicate seed {seed} for {condition}")
        rows = record.get("rows")
        _require(isinstance(rows, list) and rows, f"missing rows for {condition}/seed={seed}")
        output[seed] = rows
    return output


def _validate_alignment(result: dict[str, Any]) -> tuple[list[int], list[str], list[int]]:
    boundary = result.get("claim_boundary", {})
    _require(boundary.get("test_accessed") is False, "analysis requires test_accessed=false")
    _require(boundary.get("research_complete") is False, "analysis refuses research_complete=true packages")

    condition_maps = {condition: _rows_by_seed(result, condition) for condition in CONDITIONS}
    seeds = sorted(condition_maps[CONDITIONS[0]])
    _require(seeds, "no seeds found")
    for condition, mapping in condition_maps.items():
        _require(sorted(mapping) == seeds, f"seed mismatch for {condition}")

    canonical_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    for condition, mapping in condition_maps.items():
        for seed in seeds:
            rows = mapping[seed]
            ids = [str(row["id"]) for row in rows]
            labels = [int(row["label"]) for row in rows]
            if canonical_ids is None:
                canonical_ids = ids
                canonical_labels = labels
            else:
                _require(ids == canonical_ids, f"row identity/order mismatch for {condition}/seed={seed}")
                _require(labels == canonical_labels, f"label mismatch for {condition}/seed={seed}")
    assert canonical_ids is not None and canonical_labels is not None
    return seeds, canonical_ids, canonical_labels


def _condition_summary(result: dict[str, Any], condition: str, seeds: list[int]) -> dict[str, Any]:
    mapping = _rows_by_seed(result, condition)
    per_seed = []
    per_label_counts: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "correct": 0, "predictions": Counter()}
    )
    aggregate_predictions: Counter[int] = Counter()

    for seed in seeds:
        rows = mapping[seed]
        predictions = [int(row["prediction"]) for row in rows]
        labels = [int(row["label"]) for row in rows]
        correct = sum(int(p == y) for p, y in zip(predictions, labels, strict=True))
        histogram = Counter(predictions)
        per_seed.append(
            {
                "seed": seed,
                "rows": len(rows),
                "accuracy": correct / len(rows),
                "prediction_support": len(histogram),
                "largest_predicted_class_share": max(histogram.values()) / len(rows),
            }
        )
        aggregate_predictions.update(histogram)
        for row in rows:
            label = int(row["label"])
            prediction = int(row["prediction"])
            bucket = per_label_counts[label]
            bucket["count"] += 1
            bucket["correct"] += int(label == prediction)
            bucket["predictions"][prediction] += 1

    per_label = {}
    for label in sorted(per_label_counts):
        bucket = per_label_counts[label]
        per_label[str(label)] = {
            "count": bucket["count"],
            "accuracy": bucket["correct"] / bucket["count"],
            "prediction_histogram": {
                str(pred): count for pred, count in sorted(bucket["predictions"].items())
            },
        }

    return {
        "mean_accuracy_from_retained_rows": fmean(item["accuracy"] for item in per_seed),
        "per_seed": per_seed,
        "aggregate_prediction_histogram": {
            str(pred): count for pred, count in sorted(aggregate_predictions.items())
        },
        "per_true_label": per_label,
    }


def _pairwise_transition(
    result: dict[str, Any],
    left: str,
    right: str,
    seeds: list[int],
) -> dict[str, Any]:
    left_map = _rows_by_seed(result, left)
    right_map = _rows_by_seed(result, right)
    totals = Counter()
    per_seed = []

    for seed in seeds:
        counts = Counter()
        for left_row, right_row in zip(left_map[seed], right_map[seed], strict=True):
            label = int(left_row["label"])
            left_ok = int(left_row["prediction"]) == label
            right_ok = int(right_row["prediction"]) == label
            if not left_ok and right_ok:
                key = "fixed"
            elif left_ok and not right_ok:
                key = "regressed"
            elif left_ok and right_ok:
                key = "both_correct"
            else:
                key = "both_wrong"
            counts[key] += 1
            totals[key] += 1
        per_seed.append({"seed": seed, **{key: counts[key] for key in ("fixed", "regressed", "both_correct", "both_wrong")}})

    return {
        "left_condition": left,
        "right_condition": right,
        "interpretation": "fixed means right corrected a left error; regressed means right lost a left-correct item",
        "overall": {key: totals[key] for key in ("fixed", "regressed", "both_correct", "both_wrong")},
        "per_seed": per_seed,
    }


def _repaired_stability(result: dict[str, Any], seeds: list[int], ids: list[str]) -> dict[str, Any]:
    mapping = _rows_by_seed(result, "repaired_v5_ce")
    correct_seed_histogram: Counter[int] = Counter()
    prediction_support_histogram: Counter[int] = Counter()
    items = []

    for index, item_id in enumerate(ids):
        predictions = [int(mapping[seed][index]["prediction"]) for seed in seeds]
        label = int(mapping[seeds[0]][index]["label"])
        correct_seeds = sum(int(prediction == label) for prediction in predictions)
        support = len(set(predictions))
        correct_seed_histogram[correct_seeds] += 1
        prediction_support_histogram[support] += 1
        items.append(
            {
                "id": item_id,
                "label": label,
                "correct_seed_count": correct_seeds,
                "seed_count": len(seeds),
                "prediction_support_across_seeds": support,
                "predictions": predictions,
            }
        )

    items.sort(key=lambda item: (item["correct_seed_count"], -item["prediction_support_across_seeds"], item["id"]))
    return {
        "correct_seed_count_histogram": {
            str(key): value for key, value in sorted(correct_seed_histogram.items())
        },
        "prediction_support_histogram": {
            str(key): value for key, value in sorted(prediction_support_histogram.items())
        },
        "items_hardest_first": items,
    }


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    seeds, ids, labels = _validate_alignment(result)
    return {
        "analysis_type": "ARC_V5_NEGATIVE_RESULT_DESCRIPTIVE_SLICES",
        "source_verdict": result.get("verdict"),
        "claim_boundary": {
            "changes_frozen_protocol": False,
            "changes_decision_thresholds": False,
            "uses_confirmatory_test": False,
            "authorizes_model_selection": False,
            "research_complete": False,
            "purpose": "post-hoc mechanism/error diagnostics on already-retained validation rows",
        },
        "seeds": seeds,
        "row_count": len(ids),
        "label_histogram": {str(label): count for label, count in sorted(Counter(labels).items())},
        "conditions": {
            condition: _condition_summary(result, condition, seeds) for condition in CONDITIONS
        },
        "repaired_vs_legacy": _pairwise_transition(result, "legacy_ce", "repaired_v5_ce", seeds),
        "repaired_vs_no_quantizer": _pairwise_transition(result, "no_quantizer_ce", "repaired_v5_ce", seeds),
        "repaired_item_stability": _repaired_stability(result, seeds, ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe retained ARC-v5 negative/inconclusive validation errors without changing the frozen gate.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    analysis = analyze_result(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "analysis_type": analysis["analysis_type"],
        "source_verdict": analysis["source_verdict"],
        "seeds": analysis["seeds"],
        "row_count": analysis["row_count"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
