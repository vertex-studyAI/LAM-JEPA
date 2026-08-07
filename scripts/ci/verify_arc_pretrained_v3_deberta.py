from __future__ import annotations

import argparse
import hashlib
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
import verify_arc_pretrained_baseline as legacy


FLOAT_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(left: float, right: float, label: str, tolerance: float = FLOAT_TOLERANCE) -> None:
    require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), f"{label}: mismatch: {left!r} vs {right!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


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
    eligible = [row for row in source if len(row.choices) == required_choice_count]
    excluded = [row for row in source if len(row.choices) != required_choice_count]
    distribution = Counter(len(row.choices) for row in source)
    require(int(declared.get("source_rows", -1)) == len(source), "source row count mismatch")
    require(declared.get("source_dataset_digest") == dataset_digest(source), "source dataset digest mismatch")
    require(declared.get("source_id_digest") == id_digest(source), "source ID digest mismatch")
    require(int(declared.get("required_choice_count", -1)) == required_choice_count, "required choice count mismatch")
    require({int(k): int(v) for k, v in (declared.get("choice_count_distribution") or {}).items()} == dict(sorted(distribution.items())), "choice distribution mismatch")
    require(int(declared.get("eligible_rows", -1)) == len(eligible), "eligible row count mismatch")
    require(declared.get("eligible_dataset_digest") == dataset_digest(eligible), "eligible dataset digest mismatch")
    require(declared.get("eligible_id_digest") == id_digest(eligible), "eligible ID digest mismatch")
    require(int(declared.get("excluded_rows", -1)) == len(excluded), "excluded row count mismatch")
    require(declared.get("excluded_id_digest") == id_digest(excluded), "excluded ID digest mismatch")
    require(declared.get("excluded") == [{"id": row.item_id, "choice_count": len(row.choices)} for row in excluded], "excluded row evidence mismatch")
    used_rows = int(declared.get("used_rows", -1))
    require(0 < used_rows <= len(eligible), "invalid used-row count")
    used = eligible[:used_rows]
    require(declared.get("used_dataset_digest") == dataset_digest(used), "limits were not applied after eligibility: dataset digest mismatch")
    require(declared.get("used_id_digest") == id_digest(used), "limits were not applied after eligibility: ID digest mismatch")
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify protocol-v3 eligible frozen DeBERTa comparison.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/arc_v3_deberta_validation.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-stage", choices=["development_smoke", "validation_stage"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    records = payload.get("records")
    aggregate = payload.get("summary")

    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen scientific protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "scientific protocol is not frozen")
    require(config.get("protocol_id") == frozen["protocol_id"], "implementation config protocol mismatch")
    require(config.get("status") == "FROZEN_BEFORE_VALIDATION_EXECUTION", "implementation config is not frozen")
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol mismatch")
    require(protocol.get("implementation_config_id") == config.get("config_id"), "result implementation config id mismatch")
    require(protocol.get("implementation_config_sha256") == file_sha256(args.config), "result implementation config digest mismatch")
    require(protocol.get("run_stage") == args.expected_stage, "run stage mismatch")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "dataset mismatch")
    require(isinstance(records, list) and records, "seed records missing")
    require(isinstance(aggregate, dict), "aggregate summary missing")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this validation-stage command", "locked-test boundary weakened")
    require(protocol.get("train_validation_overlap") == 0, "used train/validation overlap")
    require(protocol.get("primary_metric") == "multiple-choice accuracy", "primary metric drift")
    require(protocol.get("robustness_check") == "deterministic reversal of answer-choice order with label remapping", "robustness contract drift")

    frozen_eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(frozen_eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 choice-count drift")
    require(protocol.get("eligibility_rule") == frozen_eligibility.get("rule") == config.get("eligibility_rule"), "eligibility rule mismatch")
    require(int(protocol.get("required_choice_count", -1)) == required_choice_count, "result required choice count mismatch")
    train_eligible = independently_verify_source(args.train, protocol.get("train_source_eligibility"), required_choice_count=required_choice_count)
    validation_eligible = independently_verify_source(args.validation, protocol.get("validation_source_eligibility"), required_choice_count=required_choice_count)
    require(not ({row.item_id for row in train_eligible} & {row.item_id for row in validation_eligible}), "eligible train/validation leakage")
    train_n = int(protocol.get("train_examples", 0))
    validation_n = int(protocol.get("validation_examples", 0))
    require(protocol.get("train_digest") == dataset_digest(train_eligible[:train_n]), "used train digest mismatch")
    require(protocol.get("train_id_digest") == id_digest(train_eligible[:train_n]), "used train ID digest mismatch")
    require(protocol.get("validation_digest") == dataset_digest(validation_eligible[:validation_n]), "used validation digest mismatch")
    require(protocol.get("validation_id_digest") == id_digest(validation_eligible[:validation_n]), "used validation ID digest mismatch")

    strong = ((frozen.get("models") or {}).get("strong_pretrained_baseline") or {})
    model_config = config.get("pretrained_model") or {}
    require(protocol.get("pretrained_model_id") == strong.get("model") == model_config.get("model_id") == "microsoft/deberta-v3-xsmall", "DeBERTa model id mismatch")
    require(protocol.get("pretrained_model_revision") == strong.get("revision") == model_config.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "DeBERTa revision mismatch")
    require(protocol.get("resolved_pretrained_revision") == strong.get("revision"), "resolved DeBERTa revision mismatch")
    require(protocol.get("pretrained_model_license") == strong.get("license") == model_config.get("license") == "MIT", "DeBERTa license mismatch")
    require(protocol.get("transformers_version") == protocol.get("transformers_version_pin") == model_config.get("transformers_version") == "4.57.6", "transformers pin mismatch")
    require(int(protocol.get("pretrained_model_trainable_parameters", 0)) > 10_000_000, "pretrained parameter count implausibly small")
    require("capacity and compute are not matched" in str(protocol.get("comparison_type", "")), "non-matched comparison boundary missing")
    require(float(protocol.get("total_wall_clock_seconds", 0.0)) > 0.0, "total wall-clock evidence missing")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 1 and len(seeds) == len(set(seeds)), "invalid seed set")
    require(len(records) == len(seeds), "seed record count mismatch")
    lam_lr = float(protocol.get("lam_jepa_learning_rate", 0.0))
    pretrained_lr = float(protocol.get("pretrained_learning_rate", 0.0))
    close(lam_lr, float(config.get("lam_jepa_learning_rate", -1.0)), "LAM learning rate", tolerance=1e-12)
    close(pretrained_lr, float(config.get("pretrained_learning_rate", -1.0)), "pretrained learning rate", tolerance=1e-12)
    require(not math.isclose(lam_lr, pretrained_lr, rel_tol=1e-12, abs_tol=1e-12), "LAM and DeBERTa learning rates were incorrectly collapsed")
    require(int(protocol.get("max_length", 0)) == int(config.get("max_length", 0)) == 96, "max_length drift")
    require(int(protocol.get("model_steps", 0)) == int(config.get("model_steps", 0)) == 1, "planner-step drift")

    if args.expected_stage == "validation_stage":
        budget = frozen.get("training_budget") or {}
        require(seeds == config.get("seeds") == budget.get("training_seeds") == [1, 2, 3, 4, 5], "validation-stage seed budget drift")
        require(int(protocol.get("epochs", 0)) == int(config.get("epochs", 0)) == int(budget.get("epochs", 0)) == 20, "validation-stage epoch budget drift")
        require(int(protocol.get("batch_size", 0)) == int(config.get("batch_size", 0)) == int(budget.get("batch_size", 0)) == 32, "validation-stage batch-size drift")
        close(lam_lr, float(budget.get("lam_jepa_learning_rate", -1.0)), "validation-stage LAM LR", tolerance=1e-12)
        close(pretrained_lr, float(budget.get("pretrained_baseline_learning_rate", -1.0)), "validation-stage DeBERTa LR", tolerance=1e-12)
        require(protocol.get("max_train_steps") is None and config.get("max_train_steps") is None, "validation-stage max-train-steps must be null")
        require(train_n == len(train_eligible) == 1117, "validation stage did not use all eligible train rows")
        require(validation_n == len(validation_eligible) == 295, "validation stage did not use all eligible validation rows")
        expected_steps = math.ceil(train_n / int(protocol["batch_size"])) * int(protocol["epochs"])
        require(int(protocol.get("expected_training_steps_per_seed", -1)) == expected_steps == 700, "validation-stage expected optimizer steps mismatch")

    canonical_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    canonical_reversed_labels: list[int] | None = None
    lam_values: list[float] = []
    pretrained_values: list[float] = []
    deltas: list[float] = []
    declared_steps: list[int] = []

    for expected_seed, record in zip(seeds, records, strict=True):
        require(record.get("seed") == expected_seed, "seed record mismatch")
        pretrained = record.get("pretrained_baseline") or {}
        lam = record.get("lam_jepa") or {}
        pretrained_metrics, ids, labels = legacy.metrics_from_rows(pretrained.get("predictions"), canonical_ids, f"seed {expected_seed}/deberta")
        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            require(labels == canonical_labels, f"seed {expected_seed}: labels changed across seeds")
        lam_metrics, lam_ids, lam_labels = legacy.metrics_from_rows(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        require(lam_ids == canonical_ids and lam_labels == labels, f"seed {expected_seed}: cross-model row/label mismatch")
        require(len(ids) == validation_n, f"seed {expected_seed}: validation row count mismatch")

        pretrained_rev_metrics, pre_rev_ids, pre_rev_labels = legacy.metrics_from_rows(pretrained.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/deberta-reversed")
        lam_rev_metrics, lam_rev_ids, lam_rev_labels = legacy.metrics_from_rows(lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed")
        require(pre_rev_ids == lam_rev_ids == canonical_ids, f"seed {expected_seed}: choice reversal changed item identity")
        require(pre_rev_labels == lam_rev_labels == [3 - label for label in labels], f"seed {expected_seed}: choice-reversal labels incorrect")
        if canonical_reversed_labels is None:
            canonical_reversed_labels = pre_rev_labels
        else:
            require(pre_rev_labels == canonical_reversed_labels, f"seed {expected_seed}: reversed labels changed across seeds")

        legacy.verify_metrics(pretrained.get("metrics"), pretrained_metrics, f"seed {expected_seed}/deberta")
        legacy.verify_metrics(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        legacy.verify_metrics(pretrained.get("choice_reversal_metrics"), pretrained_rev_metrics, f"seed {expected_seed}/deberta-reversed")
        legacy.verify_metrics(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {expected_seed}/lam-reversed")
        steps = int(pretrained.get("training_steps_executed", 0))
        require(steps >= 1, f"seed {expected_seed}: DeBERTa executed zero training steps")
        require(float(pretrained.get("training_wall_clock_seconds", 0.0)) > 0.0, f"seed {expected_seed}: DeBERTa wall-clock evidence missing")
        require(float(lam.get("training_wall_clock_seconds", 0.0)) > 0.0, f"seed {expected_seed}: LAM wall-clock evidence missing")
        if args.expected_stage == "validation_stage":
            require(steps == 700, f"seed {expected_seed}: DeBERTa did not execute frozen 700 optimization steps")
        declared_steps.append(steps)
        delta = float(lam_metrics["accuracy"] - pretrained_metrics["accuracy"])
        close(float(record.get("accuracy_delta_lam_minus_pretrained")), delta, f"seed {expected_seed}: paired accuracy delta")
        lam_values.append(lam_metrics["accuracy"])
        pretrained_values.append(pretrained_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(aggregate.get("lam_accuracy"), summary(lam_values), "lam_accuracy")
    verify_summary(aggregate.get("pretrained_accuracy"), summary(pretrained_values), "deberta_accuracy")
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_pretrained"), summary(deltas), "paired_accuracy_delta")

    report = {
        "verdict": "PROTOCOL_V3_DEBERTA_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "implementation_config_id": config["config_id"],
        "run_stage": args.expected_stage,
        "model_id": strong["model"],
        "model_revision": strong["revision"],
        "model_license": strong["license"],
        "trainable_parameters": int(protocol["pretrained_model_trainable_parameters"]),
        "seeds": seeds,
        "training_steps_by_seed": declared_steps,
        "train_source_rows": len(load_arc_split(args.train)),
        "train_eligible_rows": len(train_eligible),
        "train_used_rows": train_n,
        "validation_source_rows": len(load_arc_split(args.validation)),
        "validation_eligible_rows": len(validation_eligible),
        "validation_used_rows": validation_n,
        "lam_learning_rate": lam_lr,
        "pretrained_learning_rate": pretrained_lr,
        "max_length": int(protocol["max_length"]),
        "lam_accuracy": summary(lam_values),
        "deberta_accuracy": summary(pretrained_values),
        "paired_accuracy_delta_lam_minus_deberta": summary(deltas),
        "locked_test_evaluated": False,
        "compute_matched": False,
        "independent_reproduction": False,
        "research_complete": False,
        "floating_point_note": "Runner aggregates Torch float32 metrics; verifier recomputes exact row metrics and uses 1e-6 tolerance for numeric summaries while identities, labels, budgets, revisions, and eligibility remain exact.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
