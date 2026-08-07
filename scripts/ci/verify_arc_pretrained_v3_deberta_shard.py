from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "ci"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import verify_arc_pretrained_baseline as legacy
import verify_arc_pretrained_v3_deberta as full_verify
from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(left: float, right: float, label: str, tolerance: float = 1e-6) -> None:
    require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), f"{label}: mismatch: {left!r} vs {right!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one frozen protocol-v3 DeBERTa validation seed shard.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/arc_v3_deberta_validation.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    records = payload.get("records")

    require(args.expected_seed in [1, 2, 3, 4, 5], "expected seed is outside the frozen validation set")
    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "scientific protocol is not frozen")
    require(config.get("config_id") == "arc-v3-deberta-validation-20260807", "unexpected implementation config")
    require(config.get("status") == "FROZEN_BEFORE_VALIDATION_EXECUTION", "implementation config is not frozen")
    require(config.get("seeds") == [1, 2, 3, 4, 5], "frozen implementation seed set drift")
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol mismatch")
    require(protocol.get("implementation_config_id") == config["config_id"], "result config id mismatch")
    require(protocol.get("implementation_config_sha256") == file_sha256(args.config), "result config digest mismatch")
    require(protocol.get("run_stage") == "validation_stage", "shard is not validation-stage execution")
    require(protocol.get("frozen_config_seeds") == [1, 2, 3, 4, 5], "shard did not retain frozen five-seed contract")
    require(protocol.get("validation_shard_seed") == args.expected_seed, "shard seed marker mismatch")
    require(protocol.get("seeds") == [args.expected_seed], "shard executed an unexpected seed set")
    require(isinstance(records, list) and len(records) == 1 and records[0].get("seed") == args.expected_seed, "shard record mismatch")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this validation-stage command", "locked-test boundary weakened")
    require(protocol.get("train_validation_overlap") == 0, "used train/validation overlap")

    eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 choice-count drift")
    require(protocol.get("eligibility_rule") == eligibility.get("rule") == config.get("eligibility_rule"), "eligibility rule mismatch")
    train_eligible = full_verify.independently_verify_source(args.train, protocol.get("train_source_eligibility"), required_choice_count=required_choice_count)
    validation_eligible = full_verify.independently_verify_source(args.validation, protocol.get("validation_source_eligibility"), required_choice_count=required_choice_count)
    require(not ({row.item_id for row in train_eligible} & {row.item_id for row in validation_eligible}), "eligible train/validation leakage")
    require(int(protocol.get("train_examples", 0)) == len(train_eligible) == 1117, "shard did not use all eligible train rows")
    require(int(protocol.get("validation_examples", 0)) == len(validation_eligible) == 295, "shard did not use all eligible validation rows")
    require(protocol.get("train_digest") == dataset_digest(train_eligible), "shard train digest mismatch")
    require(protocol.get("train_id_digest") == id_digest(train_eligible), "shard train ID digest mismatch")
    require(protocol.get("validation_digest") == dataset_digest(validation_eligible), "shard validation digest mismatch")
    require(protocol.get("validation_id_digest") == id_digest(validation_eligible), "shard validation ID digest mismatch")

    budget = frozen.get("training_budget") or {}
    require(int(protocol.get("epochs", 0)) == int(config.get("epochs", 0)) == int(budget.get("epochs", 0)) == 20, "epoch budget drift")
    require(int(protocol.get("batch_size", 0)) == int(config.get("batch_size", 0)) == int(budget.get("batch_size", 0)) == 32, "batch-size budget drift")
    close(float(protocol.get("lam_jepa_learning_rate", 0.0)), float(config.get("lam_jepa_learning_rate", -1.0)), "LAM LR", tolerance=1e-12)
    close(float(protocol.get("pretrained_learning_rate", 0.0)), float(config.get("pretrained_learning_rate", -1.0)), "DeBERTa LR", tolerance=1e-12)
    close(float(protocol.get("lam_jepa_learning_rate", 0.0)), float(budget.get("lam_jepa_learning_rate", -1.0)), "frozen LAM LR", tolerance=1e-12)
    close(float(protocol.get("pretrained_learning_rate", 0.0)), float(budget.get("pretrained_baseline_learning_rate", -1.0)), "frozen DeBERTa LR", tolerance=1e-12)
    require(protocol.get("max_train_steps") is None and config.get("max_train_steps") is None, "validation shard max-train-steps must be null")
    require(int(protocol.get("model_steps", 0)) == int(config.get("model_steps", 0)) == int(budget.get("model_steps", 0)) == 1, "planner-step drift")
    require(int(protocol.get("max_length", 0)) == int(config.get("max_length", 0)) == 96, "max-length drift")
    require(int(protocol.get("expected_training_steps_per_seed", -1)) == 700, "expected optimizer-step count drift")

    strong = ((frozen.get("models") or {}).get("strong_pretrained_baseline") or {})
    model_cfg = config.get("pretrained_model") or {}
    require(protocol.get("pretrained_model_id") == strong.get("model") == model_cfg.get("model_id") == "microsoft/deberta-v3-xsmall", "model id mismatch")
    require(protocol.get("pretrained_model_revision") == protocol.get("resolved_pretrained_revision") == strong.get("revision") == model_cfg.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "model revision mismatch")
    require(protocol.get("pretrained_model_license") == strong.get("license") == model_cfg.get("license") == "MIT", "model license mismatch")
    require(protocol.get("transformers_version") == protocol.get("transformers_version_pin") == model_cfg.get("transformers_version") == "4.57.6", "transformers pin mismatch")
    require(int(protocol.get("pretrained_model_trainable_parameters", 0)) == 70830337, "DeBERTa parameter count mismatch")
    require(float(protocol.get("total_wall_clock_seconds", 0.0)) > 0.0, "shard wall-clock evidence missing")

    record = records[0]
    pretrained = record.get("pretrained_baseline") or {}
    lam = record.get("lam_jepa") or {}
    require(int(pretrained.get("training_steps_executed", 0)) == 700, "DeBERTa did not execute 700 optimizer steps")
    require(float(pretrained.get("training_wall_clock_seconds", 0.0)) > 0.0, "DeBERTa wall-clock evidence missing")
    require(float(lam.get("training_wall_clock_seconds", 0.0)) > 0.0, "LAM wall-clock evidence missing")

    pretrained_metrics, ids, labels = legacy.metrics_from_rows(pretrained.get("predictions"), None, f"seed {args.expected_seed}/deberta")
    lam_metrics, lam_ids, lam_labels = legacy.metrics_from_rows(lam.get("predictions"), ids, f"seed {args.expected_seed}/lam")
    require(ids == lam_ids and labels == lam_labels and len(ids) == 295, "cross-model validation row/label mismatch")
    pre_rev_metrics, pre_rev_ids, pre_rev_labels = legacy.metrics_from_rows(pretrained.get("choice_reversal_predictions"), ids, f"seed {args.expected_seed}/deberta-reversed")
    lam_rev_metrics, lam_rev_ids, lam_rev_labels = legacy.metrics_from_rows(lam.get("choice_reversal_predictions"), ids, f"seed {args.expected_seed}/lam-reversed")
    require(pre_rev_ids == lam_rev_ids == ids, "choice reversal changed item identity")
    require(pre_rev_labels == lam_rev_labels == [3 - label for label in labels], "choice-reversal label remapping failed")
    legacy.verify_metrics(pretrained.get("metrics"), pretrained_metrics, f"seed {args.expected_seed}/deberta")
    legacy.verify_metrics(lam.get("metrics"), lam_metrics, f"seed {args.expected_seed}/lam")
    legacy.verify_metrics(pretrained.get("choice_reversal_metrics"), pre_rev_metrics, f"seed {args.expected_seed}/deberta-reversed")
    legacy.verify_metrics(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {args.expected_seed}/lam-reversed")
    delta = float(lam_metrics["accuracy"] - pretrained_metrics["accuracy"])
    close(float(record.get("accuracy_delta_lam_minus_pretrained")), delta, "paired accuracy delta")

    report = {
        "verdict": "PROTOCOL_V3_DEBERTA_VALIDATION_SHARD_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "implementation_config_id": config["config_id"],
        "seed": args.expected_seed,
        "training_steps": 700,
        "train_eligible_rows": 1117,
        "validation_eligible_rows": 295,
        "lam_accuracy": pretrained_metrics["accuracy"] + delta,
        "deberta_accuracy": pretrained_metrics["accuracy"],
        "accuracy_delta_lam_minus_deberta": delta,
        "locked_test_evaluated": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
