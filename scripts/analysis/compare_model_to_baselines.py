from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REFERENCE_FIELDS = (
    "majority_accuracy",
    "uniform_observed_label_accuracy",
    "uniform_full_vocab_accuracy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_payload(path: Path, label: str) -> dict:
    require(path.is_file(), f"{label} output not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} payload must be a dictionary")
    return payload


def finite_probability(value: object, label: str) -> float:
    require(isinstance(value, (int, float)), f"{label} must be numeric")
    normalized = float(value)
    require(math.isfinite(normalized), f"{label} must be finite")
    require(0.0 <= normalized <= 1.0, f"{label} must be within [0, 1]")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model accuracy with label-distribution references on exactly matched evaluation rows."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/model_baseline_comparison.json"))
    args = parser.parse_args()

    model = read_payload(args.model, "model evaluation")
    baselines = read_payload(args.baselines, "baseline evaluation")

    for field in ("seed", "batch_size", "batches", "tasks"):
        require(model.get(field) == baselines.get(field), f"protocol mismatch for {field}")

    tasks = model.get("tasks")
    model_scores = model.get("scores")
    baseline_scores = baselines.get("scores")
    require(isinstance(tasks, list) and tasks, "task list must be non-empty")
    require(isinstance(model_scores, dict), "model scores must be a dictionary")
    require(isinstance(baseline_scores, dict), "baseline scores must be a dictionary")
    require(set(model_scores) == set(tasks), "model scores do not match the declared tasks")
    require(set(baseline_scores) == set(tasks), "baseline scores do not match the declared tasks")

    comparisons: dict[str, dict[str, object]] = {}
    above_majority = 0
    equal_majority = 0
    below_majority = 0

    for task in tasks:
        model_result = model_scores[task]
        baseline_result = baseline_scores[task]
        require(isinstance(model_result, dict), f"{task}: model result must be a dictionary")
        require(isinstance(baseline_result, dict), f"{task}: baseline result must be a dictionary")
        require(model_result.get("n") == baseline_result.get("n"), f"{task}: sample-count mismatch")
        require(
            model_result.get("target_semantics") == baseline_result.get("target_semantics"),
            f"{task}: target-semantics mismatch",
        )
        model_digest = model_result.get("sample_digest")
        baseline_digest = baseline_result.get("sample_digest")
        require(isinstance(model_digest, str) and len(model_digest) == 64, f"{task}: invalid model sample digest")
        require(
            model_digest == baseline_digest,
            f"{task}: sample digests differ; comparison is not paired and must not be reported",
        )

        accuracy = finite_probability(model_result.get("accuracy"), f"{task}: model accuracy")
        references = {
            field: finite_probability(baseline_result.get(field), f"{task}: {field}")
            for field in REFERENCE_FIELDS
        }
        deltas = {
            "accuracy_minus_majority": accuracy - references["majority_accuracy"],
            "accuracy_minus_uniform_observed": accuracy - references["uniform_observed_label_accuracy"],
            "accuracy_minus_uniform_full_vocab": accuracy - references["uniform_full_vocab_accuracy"],
        }

        majority_delta = deltas["accuracy_minus_majority"]
        if math.isclose(majority_delta, 0.0, rel_tol=1e-12, abs_tol=1e-12):
            relation = "equal_to_majority_reference"
            equal_majority += 1
        elif majority_delta > 0:
            relation = "above_majority_reference"
            above_majority += 1
        else:
            relation = "below_majority_reference"
            below_majority += 1

        comparisons[task] = {
            "n": model_result["n"],
            "sample_digest": model_digest,
            "target_semantics": model_result["target_semantics"],
            "model_accuracy": accuracy,
            **references,
            **deltas,
            "relation_to_majority_reference": relation,
        }

    output = {
        "checkpoint": model.get("checkpoint"),
        "checkpoint_step": model.get("checkpoint_step"),
        "protocol": {
            "seed": model["seed"],
            "batch_size": model["batch_size"],
            "batches": model["batches"],
            "tasks": tasks,
            "pairing": "exact ordered input-and-label digest match per task",
        },
        "summary": {
            "tasks": len(tasks),
            "above_majority_reference": above_majority,
            "equal_to_majority_reference": equal_majority,
            "below_majority_reference": below_majority,
        },
        "comparisons": comparisons,
        "claim_boundary": (
            "Descriptive deltas on one exactly paired sampled evaluation only. They are not confidence intervals, "
            "statistical significance, held-out generalization, answer correctness for concept-proxy tasks, "
            "educational effectiveness, benchmark validity, novelty, or superiority claims."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
