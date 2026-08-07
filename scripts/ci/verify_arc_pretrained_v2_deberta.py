from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from verify_arc_pretrained_baseline import metrics_from_rows, require, verify_metrics, verify_summary, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen protocol-v2 DeBERTa ARC development smoke.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.results.is_file(), f"results missing: {args.results}")
    require(args.protocol.is_file(), f"protocol missing: {args.protocol}")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(isinstance(payload, dict) and isinstance(frozen, dict), "results/protocol must be objects")
    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v2", "wrong frozen protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v2 is not frozen")

    strong = ((frozen.get("models") or {}).get("strong_pretrained_baseline") or {})
    expected_model = strong.get("model")
    expected_revision = strong.get("revision")
    expected_license = strong.get("license")
    require(expected_model == "microsoft/deberta-v3-xsmall", "unexpected v2 pretrained model")
    require(isinstance(expected_revision, str) and len(expected_revision) == 40, "invalid v2 model revision")
    require(expected_license == "MIT", "unexpected v2 model license")

    protocol = payload.get("protocol")
    records = payload.get("records")
    aggregate = payload.get("summary")
    require(isinstance(protocol, dict), "result protocol missing")
    require(isinstance(records, list), "records missing")
    require(isinstance(aggregate, dict), "summary missing")

    require(protocol.get("dataset") == "AI2 ARC-Challenge", "unexpected dataset")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    require(
        protocol.get("test_split_policy") == "not downloaded or evaluated by this development command",
        "locked-test boundary weakened",
    )
    for key in ("train_digest", "validation_digest", "train_id_digest", "validation_id_digest"):
        value = protocol.get(key)
        require(isinstance(value, str) and len(value) == 64, f"invalid {key}")
    require(protocol["train_digest"] != protocol["validation_digest"], "train/validation dataset digests collide")
    require(protocol["train_id_digest"] != protocol["validation_id_digest"], "train/validation ID digests collide")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid development seeds")
    require(len(records) == len(seeds), "seed record count mismatch")
    require(protocol.get("pretrained_model_id") == expected_model, "result model does not match protocol v2")
    require(protocol.get("pretrained_model_revision") == expected_revision, "declared model revision mismatch")
    require(protocol.get("resolved_pretrained_revision") == expected_revision, "resolved remote revision mismatch")
    require(protocol.get("pretrained_model_license") == expected_license, "model license mismatch")
    require(protocol.get("transformers_version") == protocol.get("transformers_version_pin"), "transformers runtime/pin mismatch")
    parameter_count = int(protocol.get("pretrained_model_trainable_parameters", 0))
    require(parameter_count > 10_000_000, "pretrained parameter count is implausibly small")
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
    require(isinstance(declared_steps, list) and len(declared_steps) == len(seeds), "training-step evidence mismatch")
    require(all(isinstance(step, int) and step >= 1 for step in declared_steps), "pretrained baseline executed zero training steps")

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
        require(pretrained.get("training_steps_executed") == declared_steps[index], "training-step evidence changed")

        pre_metrics, ids, labels = metrics_from_rows(pretrained.get("predictions"), canonical_ids, f"seed {expected_seed}/deberta")
        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            require(labels == canonical_labels, f"seed {expected_seed}: labels changed across seeds")
        require(len(ids) == validation_n, f"seed {expected_seed}: validation count mismatch")
        lam_metrics, lam_ids, lam_labels = metrics_from_rows(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        require(lam_ids == canonical_ids and lam_labels == labels, f"seed {expected_seed}: cross-model row/label mismatch")

        pre_rev_metrics, pre_rev_ids, pre_rev_labels = metrics_from_rows(
            pretrained.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/deberta-reversed"
        )
        lam_rev_metrics, lam_rev_ids, lam_rev_labels = metrics_from_rows(
            lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed"
        )
        require(pre_rev_ids == canonical_ids and lam_rev_ids == canonical_ids, "choice reversal changed item identity")
        require(pre_rev_labels == lam_rev_labels, "reversed cross-model labels differ")
        require(pre_rev_labels == [3 - label for label in labels], "choice-reversal label remapping is incorrect")
        if canonical_reversed_labels is None:
            canonical_reversed_labels = pre_rev_labels
        else:
            require(pre_rev_labels == canonical_reversed_labels, "reversed labels changed across seeds")

        verify_metrics(pretrained.get("metrics"), pre_metrics, f"seed {expected_seed}/deberta")
        verify_metrics(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metrics(pretrained.get("choice_reversal_metrics"), pre_rev_metrics, f"seed {expected_seed}/deberta-reversed")
        verify_metrics(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {expected_seed}/lam-reversed")

        delta = lam_metrics["accuracy"] - pre_metrics["accuracy"]
        require(
            math.isclose(float(record.get("accuracy_delta_lam_minus_pretrained")), delta, rel_tol=1e-9, abs_tol=1e-9),
            f"seed {expected_seed}: paired accuracy delta mismatch",
        )
        lam_accuracy.append(lam_metrics["accuracy"])
        pretrained_accuracy.append(pre_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(aggregate.get("lam_accuracy"), summary(lam_accuracy), "lam_accuracy")
    verify_summary(aggregate.get("pretrained_accuracy"), summary(pretrained_accuracy), "deberta_accuracy")
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_pretrained"), summary(deltas), "paired_delta")

    report = {
        "verdict": "PROTOCOL_V2_PRETRAINED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "model_id": expected_model,
        "model_revision": expected_revision,
        "model_license": expected_license,
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
