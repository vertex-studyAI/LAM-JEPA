from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.benchmarking.evaluation_sampling import TARGET_SEMANTICS


GENERATED_PROXY_TASKS = {"gsm8k", "reading", "tutoring", "reasoning"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the retained all-task evaluation smoke report.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.evaluation.is_file(), f"evaluation output not found: {args.evaluation}")
    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "evaluation payload must be a dictionary")

    tasks = payload.get("tasks")
    scores = payload.get("scores")
    batch_size = payload.get("batch_size")
    batches = payload.get("batches")
    checkpoint_step = payload.get("checkpoint_step")

    require(tasks == list(EDTECH_TASKS), "evaluation task order does not match EDTECH_TASKS")
    require(isinstance(scores, dict), "evaluation scores must be a dictionary")
    require(set(scores) == set(EDTECH_TASKS), "evaluation scores do not cover every declared task")
    require(isinstance(batch_size, int) and batch_size >= 1, "batch_size must be positive")
    require(isinstance(batches, int) and batches >= 1, "batches must be positive")
    require(isinstance(checkpoint_step, int) and checkpoint_step >= 1, "checkpoint_step must be positive")

    expected_examples = batch_size * batches
    verified = {}
    for task in EDTECH_TASKS:
        metrics = scores[task]
        require(isinstance(metrics, dict), f"{task}: metrics must be a dictionary")

        accuracy = metrics.get("accuracy")
        confidence = metrics.get("confidence")
        count = metrics.get("n")
        unique_inputs = metrics.get("unique_inputs")
        unique_labels = metrics.get("unique_labels")
        unique_prompts = metrics.get("unique_prompts")
        target_semantics = metrics.get("target_semantics")

        require(
            isinstance(accuracy, (int, float)) and math.isfinite(float(accuracy)),
            f"{task}: accuracy is missing or non-finite",
        )
        require(0.0 <= float(accuracy) <= 1.0, f"{task}: accuracy is outside [0, 1]")
        require(
            isinstance(confidence, (int, float)) and math.isfinite(float(confidence)),
            f"{task}: confidence is missing or non-finite",
        )
        require(0.0 <= float(confidence) <= 1.0, f"{task}: confidence is outside [0, 1]")
        require(count == expected_examples, f"{task}: expected n={expected_examples}, received {count!r}")
        require(
            isinstance(unique_inputs, int) and 1 <= unique_inputs <= count,
            f"{task}: unique_inputs must be within [1, n]",
        )
        if expected_examples > 1:
            require(unique_inputs > 1, f"{task}: evaluation rows collapsed to one repeated input")
        require(
            isinstance(unique_labels, int) and 1 <= unique_labels <= count,
            f"{task}: unique_labels must be within [1, n]",
        )
        require(
            isinstance(unique_prompts, int) and 0 <= unique_prompts <= count,
            f"{task}: unique_prompts must be within [0, n]",
        )
        if task in GENERATED_PROXY_TASKS and expected_examples > 1:
            require(unique_prompts > 1, f"{task}: generated evaluation rows reused one prompt")
        require(
            target_semantics == TARGET_SEMANTICS[task],
            f"{task}: target semantics are missing or incorrect",
        )

        verified[task] = {
            "accuracy": float(accuracy),
            "confidence": float(confidence),
            "n": count,
            "unique_inputs": unique_inputs,
            "unique_labels": unique_labels,
            "unique_prompts": unique_prompts,
            "target_semantics": target_semantics,
        }

    report = {
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_step": checkpoint_step,
        "device": payload.get("device"),
        "seed": payload.get("seed"),
        "tasks_verified": len(verified),
        "examples_per_task": expected_examples,
        "scores": verified,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
