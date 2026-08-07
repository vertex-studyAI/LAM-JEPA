from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_arc_pretrained_baseline as base
import run_arc_pretrained_v2_deberta as deberta
from lam_jepa.benchmarking.arc_challenge import (
    _predict_lam,
    _train_lam_jepa,
    dataset_digest,
    id_digest,
    load_arc_split,
    reverse_choices,
    score_predictions,
)
from lam_jepa.benchmarking.arc_protocol import ARC_PROTOCOL_CHOICE_COUNT, select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig


PROTOCOL_ID = "lam-jepa-arc-challenge-v3"
DEFAULT_CONFIG = ROOT / "configs" / "arc_v3_deberta_validation.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def eligibility_evidence(source, used) -> dict[str, object]:
    partition = select_protocol_eligible_examples(source)
    eligible_ids = [row.item_id for row in partition.eligible]
    used_ids = [row.item_id for row in used]
    if used_ids != eligible_ids[: len(used_ids)]:
        raise RuntimeError("benchmark limits were not applied after protocol-v3 eligibility")
    return {
        "source_rows": partition.original_count,
        "source_dataset_digest": dataset_digest(source),
        "source_id_digest": id_digest(source),
        "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
        "choice_count_distribution": {str(key): value for key, value in partition.choice_count_distribution.items()},
        "eligible_rows": partition.eligible_count,
        "eligible_dataset_digest": dataset_digest(partition.eligible),
        "eligible_id_digest": partition.eligible_id_digest,
        "excluded_rows": partition.excluded_count,
        "excluded_id_digest": partition.excluded_id_digest,
        "excluded": [
            {"id": row.item_id, "choice_count": len(row.choices)}
            for row in partition.excluded
        ],
        "used_rows": len(used),
        "used_dataset_digest": dataset_digest(used),
        "used_id_digest": id_digest(used),
    }


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("DeBERTa validation config is not bound to protocol v3")
    if config.get("status") != "FROZEN_BEFORE_VALIDATION_EXECUTION":
        raise ValueError("DeBERTa validation config is not frozen")
    if config.get("eligibility_rule") != "retain a row if and only if len(choices) == 4":
        raise ValueError("DeBERTa validation eligibility rule drift")
    model = config.get("pretrained_model") or {}
    if model.get("model_id") != deberta.MODEL_ID or model.get("revision") != deberta.MODEL_REVISION or model.get("license") != deberta.MODEL_LICENSE:
        raise ValueError("DeBERTa validation model identity does not match the immutable loader")
    if model.get("transformers_version") != base.PINNED_TRANSFORMERS_VERSION:
        raise ValueError("DeBERTa validation transformers version drift")
    if int(config.get("max_length", 0)) != 96:
        raise ValueError("frozen DeBERTa max_length must remain 96")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protocol-v3 eligible LAM-JEPA vs frozen DeBERTa-v3-xsmall validation.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-stage", choices=["development_smoke", "validation_stage"], required=True)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--smoke-train-limit", type=int, default=8)
    parser.add_argument("--smoke-validation-limit", type=int, default=16)
    parser.add_argument("--smoke-max-train-steps", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    frozen_config_seeds = [int(seed) for seed in config["seeds"]]
    if frozen_config_seeds != [1, 2, 3, 4, 5]:
        raise ValueError("frozen validation config seeds drifted")

    if args.run_stage == "validation_stage":
        if args.validation_seed is None:
            seeds = list(frozen_config_seeds)
            validation_shard_seed = None
        else:
            if args.validation_seed not in frozen_config_seeds:
                parser.error(f"validation shard seed {args.validation_seed} is not in frozen seeds {frozen_config_seeds}")
            seeds = [int(args.validation_seed)]
            validation_shard_seed = int(args.validation_seed)
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        train_limit = None
        validation_limit = None
        max_train_steps = config["max_train_steps"]
    else:
        if args.validation_seed is not None:
            parser.error("--validation-seed is only valid with --run-stage validation_stage")
        seeds = [int(seed) for seed in args.smoke_seeds]
        validation_shard_seed = None
        epochs = int(args.smoke_epochs)
        batch_size = int(args.smoke_batch_size)
        train_limit = int(args.smoke_train_limit)
        validation_limit = int(args.smoke_validation_limit)
        max_train_steps = int(args.smoke_max_train_steps)
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        parser.error("seed set must be non-empty and unique")
    if min(epochs, batch_size) < 1:
        parser.error("epochs and batch size must be positive")
    if max_train_steps is not None and int(max_train_steps) < 1:
        parser.error("max train steps must be positive when set")

    lam_lr = float(config["lam_jepa_learning_rate"])
    pretrained_lr = float(config["pretrained_learning_rate"])
    model_steps = int(config["model_steps"])
    max_length = int(config["max_length"])

    train_source = load_arc_split(args.train)
    validation_source = load_arc_split(args.validation)
    train_partition = select_protocol_eligible_examples(train_source)
    validation_partition = select_protocol_eligible_examples(validation_source)
    train_eligible = list(train_partition.eligible)
    validation_eligible = list(validation_partition.eligible)
    train = train_eligible[:train_limit] if train_limit else train_eligible
    validation = validation_eligible[:validation_limit] if validation_limit else validation_eligible
    if not train or not validation:
        parser.error("eligible train and validation must be non-empty")
    if any(len(row.choices) != ARC_PROTOCOL_CHOICE_COUNT for row in train + validation):
        raise RuntimeError("protocol-v3 eligibility admitted non-four-choice rows")
    overlap = {row.item_id for row in train_eligible} & {row.item_id for row in validation_eligible}
    if overlap:
        raise RuntimeError(f"eligible train/validation leakage: {sorted(overlap)[:5]}")

    train_evidence = eligibility_evidence(train_source, train)
    validation_evidence = eligibility_evidence(validation_source, validation)

    base.MODEL_ID = deberta.MODEL_ID
    base.MODEL_REVISION = deberta.MODEL_REVISION
    base.MODEL_LICENSE = deberta.MODEL_LICENSE
    base.load_pretrained = deberta.load_deberta
    resolved_revision = base.verify_remote_revision()
    if resolved_revision != deberta.MODEL_REVISION:
        raise RuntimeError("resolved DeBERTa revision changed")

    cfg = LAMJEPAConfig()
    reversed_validation = [reverse_choices(row) for row in validation]
    records: list[dict[str, object]] = []
    lam_accuracies: list[float] = []
    pretrained_accuracies: list[float] = []
    deltas: list[float] = []
    parameter_count: int | None = None
    total_started = time.perf_counter()

    for seed in seeds:
        pretrained_started = time.perf_counter()
        tokenizer, pretrained, training_steps = base.train_pretrained(
            train,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=pretrained_lr,
            max_length=max_length,
            max_train_steps=max_train_steps,
            device=args.device,
        )
        pretrained_seconds = time.perf_counter() - pretrained_started
        current_count = base.trainable_parameters(pretrained)
        if parameter_count is None:
            parameter_count = current_count
        elif current_count != parameter_count:
            raise RuntimeError("DeBERTa trainable parameter count changed across seeds")

        lam_started = time.perf_counter()
        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lam_lr,
            model_steps=model_steps,
            device=args.device,
        )
        lam_seconds = time.perf_counter() - lam_started

        pretrained_probs, pretrained_labels, pretrained_rows = base.predict_pretrained(
            tokenizer,
            pretrained,
            validation,
            batch_size=batch_size,
            max_length=max_length,
            device=args.device,
        )
        lam_probs, lam_labels, lam_rows = _predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=args.device,
        )
        if not torch.equal(pretrained_labels, lam_labels):
            raise RuntimeError("LAM-JEPA and DeBERTa evaluated different eligible validation labels")

        pretrained_rev_probs, pretrained_rev_labels, pretrained_rev_rows = base.predict_pretrained(
            tokenizer,
            pretrained,
            reversed_validation,
            batch_size=batch_size,
            max_length=max_length,
            device=args.device,
        )
        lam_rev_probs, lam_rev_labels, lam_rev_rows = _predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=args.device,
        )
        if not torch.equal(pretrained_rev_labels, lam_rev_labels):
            raise RuntimeError("LAM-JEPA and DeBERTa evaluated different reversed labels")

        pretrained_metrics = score_predictions(pretrained_probs, pretrained_labels)
        lam_metrics = score_predictions(lam_probs, lam_labels)
        pretrained_rev_metrics = score_predictions(pretrained_rev_probs, pretrained_rev_labels)
        lam_rev_metrics = score_predictions(lam_rev_probs, lam_rev_labels)
        delta = float(lam_metrics["accuracy"] - pretrained_metrics["accuracy"])
        pretrained_accuracies.append(float(pretrained_metrics["accuracy"]))
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        deltas.append(delta)
        records.append(
            {
                "seed": seed,
                "pretrained_baseline": {
                    "metrics": pretrained_metrics,
                    "choice_reversal_metrics": pretrained_rev_metrics,
                    "predictions": pretrained_rows,
                    "choice_reversal_predictions": pretrained_rev_rows,
                    "training_steps_executed": int(training_steps),
                    "training_wall_clock_seconds": pretrained_seconds,
                },
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_rev_metrics,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_rev_rows,
                    "training_wall_clock_seconds": lam_seconds,
                },
                "accuracy_delta_lam_minus_pretrained": delta,
            }
        )
        del pretrained, tokenizer, lam

    assert parameter_count is not None
    total_seconds = time.perf_counter() - total_started
    expected_steps = math.ceil(len(train) / batch_size) * epochs if max_train_steps is None else min(int(max_train_steps), math.ceil(len(train) / batch_size) * epochs)
    payload = {
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "implementation_config_id": config["config_id"],
            "implementation_config_sha256": file_sha256(args.config),
            "run_stage": args.run_stage,
            "frozen_config_seeds": frozen_config_seeds,
            "validation_shard_seed": validation_shard_seed,
            "dataset": "AI2 ARC-Challenge",
            "eligibility_rule": config["eligibility_rule"],
            "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
            "train_source_eligibility": train_evidence,
            "validation_source_eligibility": validation_evidence,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lam_jepa_learning_rate": lam_lr,
            "pretrained_learning_rate": pretrained_lr,
            "max_length": max_length,
            "max_train_steps": max_train_steps,
            "model_steps": model_steps,
            "optimizer": config["optimizer"],
            "device": args.device,
            "primary_metric": "multiple-choice accuracy",
            "calibration_metrics": ["brier", "ece", "mean_true_class_probability"],
            "robustness_check": "deterministic reversal of answer-choice order with label remapping",
            "pretrained_model_id": deberta.MODEL_ID,
            "pretrained_model_revision": deberta.MODEL_REVISION,
            "resolved_pretrained_revision": resolved_revision,
            "pretrained_model_license": deberta.MODEL_LICENSE,
            "pretrained_model_trainable_parameters": parameter_count,
            "transformers_version": base.transformers.__version__,
            "transformers_version_pin": base.PINNED_TRANSFORMERS_VERSION,
            "expected_training_steps_per_seed": expected_steps,
            "comparison_type": "strong pretrained language-model baseline; capacity and compute are not matched to LAM-JEPA",
            "test_split_policy": "not downloaded or evaluated by this validation-stage command",
            "total_wall_clock_seconds": total_seconds,
            "claim_boundary": (
                "Protocol-v3 validation-stage strong-baseline execution only. It is not a locked-test result, not compute-matched, "
                "not independent reproduction, and cannot by itself authorize superiority, external validation, or RESEARCH_COMPLETE."
            ),
        },
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_accuracies),
            "pretrained_accuracy": summarize(pretrained_accuracies),
            "paired_accuracy_delta_lam_minus_pretrained": summarize(deltas),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
