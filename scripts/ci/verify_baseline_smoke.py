from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.benchmarking.evaluation_sampling import TARGET_SEMANTICS


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LAM-JEPA label-baseline smoke evidence.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    scores = payload.get("scores")
    batch_size = payload.get("batch_size")
    batches = payload.get("batches")
    vocab_size = payload.get("vocab_size")

    if tasks != list(EDTECH_TASKS):
        raise SystemExit(f"task registry mismatch: {tasks!r}")
    if not isinstance(scores, dict) or set(scores) != set(EDTECH_TASKS):
        raise SystemExit("baseline scores do not cover every declared task")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise SystemExit("invalid batch_size")
    if not isinstance(batches, int) or batches < 1:
        raise SystemExit("invalid batches")
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise SystemExit("invalid vocab_size")

    expected_n = batch_size * batches
    verified: dict[str, dict[str, float | int | str | bool]] = {}
    for task in EDTECH_TASKS:
        result = scores[task]
        required = {
            "n",
            "unique_inputs",
            "unique_labels",
            "unique_prompts",
            "majority_label",
            "majority_count",
            "majority_accuracy",
            "uniform_observed_label_accuracy",
            "uniform_full_vocab_accuracy",
            "vocab_size",
            "target_semantics",
            "baseline_semantics",
        }
        missing = required.difference(result)
        if missing:
            raise SystemExit(f"{task}: missing baseline fields {sorted(missing)}")

        n = int(result["n"])
        unique_inputs = int(result["unique_inputs"])
        unique_labels = int(result["unique_labels"])
        majority_label = int(result["majority_label"])
        majority_count = int(result["majority_count"])
        majority_accuracy = float(result["majority_accuracy"])
        uniform_observed = float(result["uniform_observed_label_accuracy"])
        uniform_full_vocab = float(result["uniform_full_vocab_accuracy"])

        if n != expected_n:
            raise SystemExit(f"{task}: expected n={expected_n}, got {n}")
        if not 1 <= unique_inputs <= n:
            raise SystemExit(f"{task}: invalid unique_inputs={unique_inputs} for n={n}")
        if not 1 <= unique_labels <= min(n, vocab_size):
            raise SystemExit(f"{task}: invalid unique_labels={unique_labels}")
        if not 0 <= majority_label < vocab_size:
            raise SystemExit(f"{task}: majority_label outside vocabulary")
        if not 1 <= majority_count <= n:
            raise SystemExit(f"{task}: invalid majority_count={majority_count}")
        for name, value in {
            "majority_accuracy": majority_accuracy,
            "uniform_observed_label_accuracy": uniform_observed,
            "uniform_full_vocab_accuracy": uniform_full_vocab,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(f"{task}: invalid {name}={value}")
        if not close(majority_accuracy, majority_count / n):
            raise SystemExit(f"{task}: majority accuracy is inconsistent with its count")
        if not close(uniform_observed, 1.0 / unique_labels):
            raise SystemExit(f"{task}: observed-uniform accuracy is inconsistent with label support")
        if not close(uniform_full_vocab, 1.0 / vocab_size):
            raise SystemExit(f"{task}: full-vocabulary accuracy is inconsistent with vocab_size")
        if majority_accuracy + 1e-12 < uniform_observed:
            raise SystemExit(f"{task}: majority reference is below observed-uniform reference")
        if int(result["vocab_size"]) != vocab_size:
            raise SystemExit(f"{task}: per-task vocab_size mismatch")
        if result["target_semantics"] != TARGET_SEMANTICS[task]:
            raise SystemExit(f"{task}: target semantics mismatch")
        if result["baseline_semantics"] != "sampled-label-distribution reference; no model executed":
            raise SystemExit(f"{task}: baseline semantics were weakened or changed")

        verified[task] = {
            "n": n,
            "unique_inputs": unique_inputs,
            "unique_labels": unique_labels,
            "majority_accuracy": majority_accuracy,
            "uniform_observed_label_accuracy": uniform_observed,
            "uniform_full_vocab_accuracy": uniform_full_vocab,
            "target_semantics": result["target_semantics"],
            "verified": True,
        }

    report = {
        "status": "passed",
        "tasks": list(EDTECH_TASKS),
        "expected_n_per_task": expected_n,
        "vocab_size": vocab_size,
        "verified": verified,
        "claim_boundary": "label-distribution references only; no model or educational-effectiveness claim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
