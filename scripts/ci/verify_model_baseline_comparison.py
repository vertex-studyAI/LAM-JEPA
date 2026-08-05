from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.benchmarking.evaluation_sampling import TARGET_SEMANTICS


REFERENCE_FIELDS = (
    "majority_accuracy",
    "uniform_observed_label_accuracy",
    "uniform_full_vocab_accuracy",
)
DELTA_FIELDS = {
    "accuracy_minus_majority": "majority_accuracy",
    "accuracy_minus_uniform_observed": "uniform_observed_label_accuracy",
    "accuracy_minus_uniform_full_vocab": "uniform_full_vocab_accuracy",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path, label: str) -> dict:
    require(path.is_file(), f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} must be a dictionary")
    return payload


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an exactly paired model-to-baseline comparison artifact.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = read_json(args.model, "model evaluation")
    baselines = read_json(args.baselines, "baseline evaluation")
    comparison = read_json(args.comparison, "comparison artifact")

    expected_tasks = list(EDTECH_TASKS)
    require(model.get("tasks") == expected_tasks, "model task registry mismatch")
    require(baselines.get("tasks") == expected_tasks, "baseline task registry mismatch")
    for field in ("seed", "batch_size", "batches"):
        require(model.get(field) == baselines.get(field), f"protocol mismatch for {field}")
        require(comparison.get("protocol", {}).get(field) == model.get(field), f"comparison protocol mismatch for {field}")
    require(comparison.get("protocol", {}).get("tasks") == expected_tasks, "comparison task registry mismatch")
    require(
        comparison.get("protocol", {}).get("pairing") == "exact ordered input-and-label digest match per task",
        "comparison pairing contract is missing or weakened",
    )

    model_scores = model.get("scores")
    baseline_scores = baselines.get("scores")
    comparisons = comparison.get("comparisons")
    require(isinstance(model_scores, dict), "model scores missing")
    require(isinstance(baseline_scores, dict), "baseline scores missing")
    require(isinstance(comparisons, dict), "comparison rows missing")
    require(set(model_scores) == set(expected_tasks), "model scores incomplete")
    require(set(baseline_scores) == set(expected_tasks), "baseline scores incomplete")
    require(set(comparisons) == set(expected_tasks), "comparison rows incomplete")

    relations = {
        "above_majority_reference": 0,
        "equal_to_majority_reference": 0,
        "below_majority_reference": 0,
    }
    verified: dict[str, dict[str, object]] = {}

    for task in expected_tasks:
        model_row = model_scores[task]
        baseline_row = baseline_scores[task]
        row = comparisons[task]
        require(isinstance(model_row, dict) and isinstance(baseline_row, dict) and isinstance(row, dict), f"{task}: invalid row")
        require(model_row.get("n") == baseline_row.get("n") == row.get("n"), f"{task}: sample count mismatch")
        require(model_row.get("target_semantics") == TARGET_SEMANTICS[task], f"{task}: model target semantics mismatch")
        require(baseline_row.get("target_semantics") == TARGET_SEMANTICS[task], f"{task}: baseline target semantics mismatch")
        require(row.get("target_semantics") == TARGET_SEMANTICS[task], f"{task}: comparison target semantics mismatch")

        model_digest = model_row.get("sample_digest")
        baseline_digest = baseline_row.get("sample_digest")
        require(isinstance(model_digest, str) and len(model_digest) == 64, f"{task}: invalid model digest")
        require(model_digest == baseline_digest == row.get("sample_digest"), f"{task}: exact sample pairing failed")

        accuracy = float(model_row["accuracy"])
        require(math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0, f"{task}: invalid model accuracy")
        require(close(float(row["model_accuracy"]), accuracy), f"{task}: model accuracy was altered")

        for field in REFERENCE_FIELDS:
            reference = float(baseline_row[field])
            require(math.isfinite(reference) and 0.0 <= reference <= 1.0, f"{task}: invalid {field}")
            require(close(float(row[field]), reference), f"{task}: {field} was altered")

        for delta_field, reference_field in DELTA_FIELDS.items():
            expected_delta = accuracy - float(baseline_row[reference_field])
            require(close(float(row[delta_field]), expected_delta), f"{task}: incorrect {delta_field}")

        majority_delta = accuracy - float(baseline_row["majority_accuracy"])
        if close(majority_delta, 0.0):
            expected_relation = "equal_to_majority_reference"
        elif majority_delta > 0:
            expected_relation = "above_majority_reference"
        else:
            expected_relation = "below_majority_reference"
        require(row.get("relation_to_majority_reference") == expected_relation, f"{task}: incorrect majority relation")
        relations[expected_relation] += 1
        verified[task] = {
            "sample_digest": model_digest,
            "target_semantics": TARGET_SEMANTICS[task],
            "model_accuracy": accuracy,
            "majority_accuracy": float(baseline_row["majority_accuracy"]),
            "verified": True,
        }

    summary = comparison.get("summary")
    require(isinstance(summary, dict), "comparison summary missing")
    require(summary.get("tasks") == len(expected_tasks), "comparison task count mismatch")
    for relation, count in relations.items():
        require(summary.get(relation) == count, f"comparison summary mismatch for {relation}")

    claim_boundary = comparison.get("claim_boundary")
    require(isinstance(claim_boundary, str), "comparison claim boundary missing")
    for required_phrase in (
        "not confidence intervals",
        "statistical significance",
        "answer correctness for concept-proxy tasks",
        "educational effectiveness",
        "benchmark validity",
        "novelty",
        "superiority claims",
    ):
        require(required_phrase in claim_boundary, f"claim boundary missing: {required_phrase}")

    report = {
        "status": "passed",
        "tasks": expected_tasks,
        "protocol": comparison["protocol"],
        "summary": summary,
        "verified": verified,
        "claim_boundary": claim_boundary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
