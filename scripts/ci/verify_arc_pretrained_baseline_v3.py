from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

PROTOCOL_ID = "lam-jepa-arc-challenge-v3"
MODEL_ID = "microsoft/deberta-v3-xsmall"
MODEL_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
MODEL_LICENSE = "MIT"
TRANSFORMERS_VERSION = "4.57.6"
SENTENCEPIECE_VERSION = "0.2.2"
TOKENIZER_MAX_LENGTH = 128


def load_helpers():
    helper_path = Path(__file__).with_name("verify_arc_pretrained_baseline.py")
    spec = importlib.util.spec_from_file_location("lam_jepa_arc_pretrained_verifier_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier helpers: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()
require = helpers.require
metrics_from_rows = helpers.metrics_from_rows
verify_metrics = helpers.verify_metrics
summary = helpers.summary
verify_summary = helpers.verify_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify protocol-v3 DeBERTa ARC baseline evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.results.is_file(), f"results missing: {args.results}")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "results must be an object")
    protocol = payload.get("protocol")
    records = payload.get("records")
    aggregate = payload.get("summary")
    require(isinstance(protocol, dict), "protocol missing")
    require(isinstance(records, list), "records missing")
    require(isinstance(aggregate, dict), "summary missing")

    require(protocol.get("frozen_protocol_id") == PROTOCOL_ID, "result is not bound to frozen ARC protocol v3")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected dataset")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    require(
        protocol.get("test_split_policy") == "not downloaded or evaluated by this development command",
        "test boundary weakened",
    )
    for key in ("train_digest", "validation_digest", "train_id_digest", "validation_id_digest"):
        value = protocol.get(key)
        require(isinstance(value, str) and len(value) == 64, f"invalid {key}")
    require(protocol["train_digest"] != protocol["validation_digest"], "dataset split digests collide")
    require(protocol["train_id_digest"] != protocol["validation_id_digest"], "ID split digests collide")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid smoke seeds")
    require(len(records) == len(seeds), "seed record count mismatch")
    require(protocol.get("pretrained_model_id") == MODEL_ID, "pretrained model id does not match protocol v3")
    require(protocol.get("pretrained_model_revision") == MODEL_REVISION, "declared pretrained revision changed")
    require(protocol.get("resolved_pretrained_revision") == MODEL_REVISION, "resolved pretrained revision changed")
    require(protocol.get("pretrained_model_license") == MODEL_LICENSE, "pretrained license mismatch")
    require(protocol.get("transformers_version") == TRANSFORMERS_VERSION, "runtime transformers version changed")
    require(protocol.get("transformers_version_pin") == TRANSFORMERS_VERSION, "transformers pin changed")
    require(protocol.get("sentencepiece_version_pin") == SENTENCEPIECE_VERSION, "sentencepiece pin changed")
    require(int(protocol.get("max_length", 0)) == TOKENIZER_MAX_LENGTH, "tokenizer max length drift")
    parameter_count = int(protocol.get("pretrained_model_trainable_parameters", 0))
    require(parameter_count > 10_000_000, "pretrained baseline parameter count is implausibly small")
    require("capacity and compute are not matched" in str(protocol.get("comparison_type", "")), "comparison boundary missing")
    require(int(protocol.get("final_seed_requirement", 0)) >= 5, "final seed requirement weakened")
    require(protocol.get("primary_metric") == "multiple-choice accuracy", "primary metric changed")
    require(
        protocol.get("robustness_check") == "deterministic reversal of answer-choice order with label remapping",
        "robustness contract changed",
    )
    claim = str(protocol.get("claim_boundary", ""))
    for phrase in (">=5-seed", "locked-test", "compute-matched", "independent reproduction", "RESEARCH_COMPLETE"):
        require(phrase in claim, f"claim boundary missing: {phrase}")

    declared_steps = protocol.get("pretrained_training_steps_by_seed")
    require(isinstance(declared_steps, list) and len(declared_steps) == len(seeds), "training-step record mismatch")
    require(all(isinstance(step, int) and step >= 1 for step in declared_steps), "pretrained baseline must execute training")

    validation_n = int(protocol.get("validation_examples", 0))
    require(validation_n > 0, "validation set is empty")
    canonical_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    canonical_reversed_labels: list[int] | None = None
    lam_accuracy: list[float] = []
    pretrained_accuracy: list[float] = []
    deltas: list[float] = []

    for index, (expected_seed, record) in enumerate(zip(seeds, records, strict=True)):
        require(isinstance(record, dict) and record.get("seed") == expected_seed, "seed record mismatch")
        pretrained = record.get("pretrained_baseline")
        lam = record.get("lam_jepa")
        require(isinstance(pretrained, dict) and isinstance(lam, dict), "model record missing")
        require(pretrained.get("training_steps_executed") == declared_steps[index], "training-step evidence mismatch")

        pre_metrics, ids, labels = metrics_from_rows(
            pretrained.get("predictions"), canonical_ids, f"seed {expected_seed}/pretrained"
        )
        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            require(labels == canonical_labels, f"seed {expected_seed}: labels changed across seeds")
        require(len(ids) == validation_n, f"seed {expected_seed}: validation count mismatch")
        lam_metrics, lam_ids, lam_labels = metrics_from_rows(
            lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam"
        )
        require(lam_ids == canonical_ids and lam_labels == labels, f"seed {expected_seed}: cross-model row mismatch")

        pre_rev_metrics, pre_rev_ids, pre_rev_labels = metrics_from_rows(
            pretrained.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/pretrained-reversed"
        )
        lam_rev_metrics, lam_rev_ids, lam_rev_labels = metrics_from_rows(
            lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed"
        )
        require(pre_rev_ids == canonical_ids and lam_rev_ids == canonical_ids, "choice reversal changed item identity")
        require(pre_rev_labels == lam_rev_labels, "reversed model labels differ")
        require(pre_rev_labels == [3 - label for label in labels], "choice-reversal label remapping incorrect")
        if canonical_reversed_labels is None:
            canonical_reversed_labels = pre_rev_labels
        else:
            require(pre_rev_labels == canonical_reversed_labels, "reversed labels changed across seeds")

        verify_metrics(pretrained.get("metrics"), pre_metrics, f"seed {expected_seed}/pretrained")
        verify_metrics(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metrics(pretrained.get("choice_reversal_metrics"), pre_rev_metrics, f"seed {expected_seed}/pretrained-reversed")
        verify_metrics(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {expected_seed}/lam-reversed")

        delta = lam_metrics["accuracy"] - pre_metrics["accuracy"]
        require(
            math.isclose(float(record.get("accuracy_delta_lam_minus_pretrained")), delta, rel_tol=1e-9, abs_tol=1e-9),
            f"seed {expected_seed}: paired delta mismatch",
        )
        lam_accuracy.append(lam_metrics["accuracy"])
        pretrained_accuracy.append(pre_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(aggregate.get("lam_accuracy"), summary(lam_accuracy), "lam_accuracy")
    verify_summary(aggregate.get("pretrained_accuracy"), summary(pretrained_accuracy), "pretrained_accuracy")
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_pretrained"), summary(deltas), "paired_delta")

    report = {
        "verdict": "PROTOCOL_V3_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "frozen_protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "transformers_version": TRANSFORMERS_VERSION,
        "sentencepiece_version": SENTENCEPIECE_VERSION,
        "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
        "trainable_parameters": parameter_count,
        "seeds": seeds,
        "validation_examples": validation_n,
        "locked_test_evaluated": False,
        "compute_matched": False,
        "independent_reproduction": False,
        "claim_boundary_preserved": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
