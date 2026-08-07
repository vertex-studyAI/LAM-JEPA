from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "ci"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split
import verify_arc_matched_baseline as legacy


FLOAT_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(left: float, right: float, label: str, tolerance: float = FLOAT_TOLERANCE) -> None:
    require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), f"{label}: mismatch: {left!r} vs {right!r}")


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(declared: object, expected: dict[str, float | int], label: str) -> None:
    require(isinstance(declared, dict), f"{label}: summary missing")
    require(int(declared.get("n", -1)) == int(expected["n"]), f"{label}: n mismatch")
    close(float(declared.get("mean", float("nan"))), float(expected["mean"]), f"{label}: mean")
    close(float(declared.get("std", float("nan"))), float(expected["std"]), f"{label}: std")


def independently_verify_source(path: Path, declared: object, *, required_choice_count: int) -> list:
    require(isinstance(declared, dict), "source eligibility evidence missing")
    source = load_arc_split(path)
    eligible = [example for example in source if len(example.choices) == required_choice_count]
    excluded = [example for example in source if len(example.choices) != required_choice_count]
    distribution = Counter(len(example.choices) for example in source)
    require(int(declared.get("source_rows", -1)) == len(source), "source row count mismatch")
    require(declared.get("source_dataset_digest") == dataset_digest(source), "source dataset digest mismatch")
    require(declared.get("source_id_digest") == id_digest(source), "source ID digest mismatch")
    require(int(declared.get("required_choice_count", -1)) == required_choice_count, "source required choice count mismatch")
    require({int(key): int(value) for key, value in (declared.get("choice_count_distribution") or {}).items()} == dict(sorted(distribution.items())), "source choice-count distribution mismatch")
    require(int(declared.get("eligible_rows", -1)) == len(eligible), "eligible row count mismatch")
    require(declared.get("eligible_dataset_digest") == dataset_digest(eligible), "eligible dataset digest mismatch")
    require(declared.get("eligible_id_digest") == id_digest(eligible), "eligible ID digest mismatch")
    require(int(declared.get("excluded_rows", -1)) == len(excluded), "excluded row count mismatch")
    require(declared.get("excluded_id_digest") == id_digest(excluded), "excluded ID digest mismatch")
    require(
        declared.get("excluded") == [{"id": example.item_id, "choice_count": len(example.choices)} for example in excluded],
        "excluded row evidence mismatch",
    )
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify protocol-v3 matched ARC execution without legacy smoke-only wording checks.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-stage", choices=["development_smoke", "validation_stage"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    records = payload.get("records")
    aggregate = payload.get("summary")

    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v3 is not frozen")
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol mismatch")
    require(protocol.get("run_stage") == args.expected_stage, "run-stage mismatch")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected dataset")
    require(isinstance(records, list) and records, "seed records missing")
    require(isinstance(aggregate, dict), "aggregate summary missing")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked-test boundary weakened")
    require(protocol.get("test_split_accessed") is False, "wrapper reports test access")
    require(protocol.get("eligibility_applied_before_limits") is True, "eligibility-before-limits evidence missing")
    require(protocol.get("train_validation_overlap") == 0, "used train/validation overlap detected")
    require(protocol.get("primary_metric") == "multiple-choice accuracy", "primary metric drift")
    require(protocol.get("robustness_check") == "deterministic reversal of answer-choice order with label remapping", "robustness contract drift")
    require(protocol.get("strong_pretrained_baseline") == "NOT_INCLUDED", "matched-only path must not masquerade as strong pretrained comparison")

    eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 choice-count drift")
    require(protocol.get("eligibility_rule") == eligibility.get("rule"), "eligibility rule mismatch")
    require(int(protocol.get("required_choice_count", -1)) == required_choice_count, "result required choice count mismatch")

    train_eligible = independently_verify_source(args.train, protocol.get("train_source_eligibility"), required_choice_count=required_choice_count)
    validation_eligible = independently_verify_source(args.validation, protocol.get("validation_source_eligibility"), required_choice_count=required_choice_count)
    require(not ({example.item_id for example in train_eligible} & {example.item_id for example in validation_eligible}), "eligible train/validation leakage")
    train_n = int(protocol.get("train_examples", 0))
    validation_n = int(protocol.get("validation_examples", 0))
    require(0 < train_n <= len(train_eligible), "invalid used train count")
    require(0 < validation_n <= len(validation_eligible), "invalid used validation count")
    used_train = train_eligible[:train_n]
    used_validation = validation_eligible[:validation_n]
    require(protocol.get("train_digest") == dataset_digest(used_train), "used train digest is not eligible-prefix digest")
    require(protocol.get("train_id_digest") == id_digest(used_train), "used train ID digest is not eligible-prefix digest")
    require(protocol.get("validation_digest") == dataset_digest(used_validation), "used validation digest is not eligible-prefix digest")
    require(protocol.get("validation_id_digest") == id_digest(used_validation), "used validation ID digest is not eligible-prefix digest")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid seed set")
    require(len(records) == len(seeds), "seed record count mismatch")

    lam_total = int(protocol.get("lam_total_trainable_parameters", 0))
    lam_active = int(protocol.get("lam_gradient_active_parameters", 0))
    matched_total = int(protocol.get("matched_supervised_trainable_parameters", 0))
    matched_active = int(protocol.get("matched_supervised_gradient_active_parameters", 0))
    require(lam_total > 0 and lam_active > 0 and matched_total > 0, "parameter evidence missing")
    require(lam_active <= lam_total, "LAM active parameters exceed total trainable")
    require(matched_active == matched_total, "matched baseline contains inactive trainable padding")
    match_basis = str(protocol.get("parameter_match_basis", ""))
    require("gradient-active" in match_basis, "matched-capacity basis is not gradient-active")
    matched_contract = ((frozen.get("models") or {}).get("matched_capacity_supervised_baseline") or {})
    lower = float(matched_contract.get("allowed_parameter_ratio_min", 0.0))
    upper = float(matched_contract.get("allowed_parameter_ratio_max", 0.0))
    ratio = matched_active / lam_active
    require(lower <= ratio <= upper, f"matched parameter ratio outside v3: {ratio:.9f}")
    declared_gap = float(protocol.get("parameter_relative_gap", -1.0))
    close(declared_gap, abs(matched_active - lam_active) / lam_active, "parameter relative gap", tolerance=1e-12)
    runner_tolerance = float(protocol.get("parameter_match_tolerance", -1.0))
    require(0.0 < runner_tolerance <= max(1.0 - lower, upper - 1.0) + 1e-12, "runner parameter tolerance weaker than frozen v3")
    require(float(protocol.get("wall_clock_seconds", 0.0)) > 0.0, "wall-clock evidence missing")
    expected_steps = math.ceil(train_n / int(protocol.get("batch_size", 0))) * int(protocol.get("epochs", 0))
    require(int(protocol.get("optimization_steps_per_model_per_seed", -1)) == expected_steps, "optimization-step evidence mismatch")

    if args.expected_stage == "validation_stage":
        budget = frozen.get("training_budget") or {}
        require(seeds == budget.get("training_seeds") == [1, 2, 3, 4, 5], "validation-stage seeds drift")
        require(int(protocol.get("epochs", 0)) == int(budget.get("epochs", 0)) == 20, "validation-stage epochs drift")
        require(int(protocol.get("batch_size", 0)) == int(budget.get("batch_size", 0)) == 32, "validation-stage batch size drift")
        close(float(protocol.get("learning_rate", 0.0)), float(budget.get("matched_baseline_learning_rate", -1.0)), "validation-stage matched LR", tolerance=1e-12)
        require(int(protocol.get("model_steps", 0)) == int(budget.get("model_steps", 0)) == 1, "validation-stage planner-step drift")
        require(train_n == len(train_eligible) == 1117, "validation-stage did not use all eligible train rows")
        require(validation_n == len(validation_eligible) == 295, "validation-stage did not use all eligible validation rows")

    canonical_ids: list[str] | None = None
    canonical_reversed_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    canonical_reversed_labels: list[int] | None = None
    lam_accuracies: list[float] = []
    matched_accuracies: list[float] = []
    deltas: list[float] = []

    for expected_seed, record in zip(seeds, records, strict=True):
        require(record.get("seed") == expected_seed, "seed record order mismatch")
        lam = record.get("lam_jepa") or {}
        matched = record.get("matched_supervised") or {}
        lam_metrics, ids = legacy.recompute_metrics(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        if canonical_ids is None:
            canonical_ids = ids
        matched_metrics, matched_ids = legacy.recompute_metrics(matched.get("predictions"), canonical_ids, f"seed {expected_seed}/matched")
        require(matched_ids == canonical_ids and len(ids) == validation_n, f"seed {expected_seed}: cross-model eligible row mismatch")
        lam_reverse_metrics, reverse_ids = legacy.recompute_metrics(lam.get("choice_reversal_predictions"), canonical_reversed_ids, f"seed {expected_seed}/lam-reversed")
        if canonical_reversed_ids is None:
            canonical_reversed_ids = reverse_ids
        matched_reverse_metrics, matched_reverse_ids = legacy.recompute_metrics(matched.get("choice_reversal_predictions"), canonical_reversed_ids, f"seed {expected_seed}/matched-reversed")
        require(matched_reverse_ids == canonical_reversed_ids == canonical_ids, f"seed {expected_seed}: choice reversal changed item identity/order")

        lam_labels = [int(row["label"]) for row in lam["predictions"]]
        matched_labels = [int(row["label"]) for row in matched["predictions"]]
        lam_reverse_labels = [int(row["label"]) for row in lam["choice_reversal_predictions"]]
        matched_reverse_labels = [int(row["label"]) for row in matched["choice_reversal_predictions"]]
        require(lam_labels == matched_labels, f"seed {expected_seed}: cross-model labels differ")
        require(lam_reverse_labels == matched_reverse_labels == [3 - label for label in lam_labels], f"seed {expected_seed}: reversed labels incorrect")
        if canonical_labels is None:
            canonical_labels = lam_labels
            canonical_reversed_labels = lam_reverse_labels
        else:
            require(lam_labels == canonical_labels and lam_reverse_labels == canonical_reversed_labels, f"seed {expected_seed}: labels changed across seeds")

        legacy.verify_metric_row(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        legacy.verify_metric_row(matched.get("metrics"), matched_metrics, f"seed {expected_seed}/matched")
        legacy.verify_metric_row(lam.get("choice_reversal_metrics"), lam_reverse_metrics, f"seed {expected_seed}/lam-reversed")
        legacy.verify_metric_row(matched.get("choice_reversal_metrics"), matched_reverse_metrics, f"seed {expected_seed}/matched-reversed")
        delta = float(lam_metrics["accuracy"] - matched_metrics["accuracy"])
        close(float(record.get("accuracy_delta_lam_minus_matched")), delta, f"seed {expected_seed}: paired accuracy delta")
        lam_accuracies.append(lam_metrics["accuracy"])
        matched_accuracies.append(matched_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(aggregate.get("lam_accuracy"), summary(lam_accuracies), "lam_accuracy")
    verify_summary(aggregate.get("matched_supervised_accuracy"), summary(matched_accuracies), "matched_supervised_accuracy")
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_matched"), summary(deltas), "paired_accuracy_delta")

    report = {
        "verdict": "PROTOCOL_V3_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "run_stage": args.expected_stage,
        "seeds": seeds,
        "train_source_rows": len(load_arc_split(args.train)),
        "train_eligible_rows": len(train_eligible),
        "train_used_rows": train_n,
        "validation_source_rows": len(load_arc_split(args.validation)),
        "validation_eligible_rows": len(validation_eligible),
        "validation_used_rows": validation_n,
        "lam_gradient_active_parameters": lam_active,
        "matched_gradient_active_parameters": matched_active,
        "parameter_ratio": ratio,
        "optimization_steps_per_model_per_seed": expected_steps,
        "lam_accuracy": summary(lam_accuracies),
        "matched_accuracy": summary(matched_accuracies),
        "paired_accuracy_delta_lam_minus_matched": summary(deltas),
        "locked_test_evaluated": False,
        "strong_pretrained_baseline_included": False,
        "research_complete": False,
        "floating_point_note": "Runner aggregates Torch float32 accuracies; verifier recomputes exact row fractions and uses 1e-6 tolerance for numeric summaries while identities, labels, budgets, digests, and parameter counts remain exact.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
