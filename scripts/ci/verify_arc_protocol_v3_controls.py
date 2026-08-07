from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def metrics_from_rows(rows: object) -> tuple[dict[str, float], list[str], list[int]]:
    require(isinstance(rows, list) and rows, "prediction rows missing")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        require(isinstance(row, dict), "prediction row must be an object")
        ids.append(str(row.get("id")))
        label = int(row.get("label", -1))
        prediction = int(row.get("prediction", -1))
        probs = row.get("probabilities")
        require(0 <= label < 4 and 0 <= prediction < 4, "invalid ARC class index")
        require(isinstance(probs, list) and len(probs) == 4, "invalid probability vector")
        probs = [float(value) for value in probs]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probs), "invalid probability value")
        require(abs(sum(probs) - 1.0) <= 1e-5, "probabilities do not sum to one")
        require(max(range(4), key=lambda index: probs[index]) == prediction, "prediction/probability mismatch")
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(probs)

    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels, strict=True)) / len(labels)
    brier = sum(
        sum((prob - (1.0 if index == label else 0.0)) ** 2 for index, prob in enumerate(probs))
        for probs, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    true_prob = statistics.fmean(probs[label] for probs, label in zip(probabilities, labels, strict=True))
    confidences = [max(probs) for probs in probabilities]
    correctness = [float(pred == label) for pred, label in zip(predictions, labels, strict=True)]
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            index
            for index, confidence in enumerate(confidences)
            if (confidence >= lower if bin_index == 0 else confidence > lower) and confidence <= upper
        ]
        if members:
            bin_acc = statistics.fmean(correctness[index] for index in members)
            bin_conf = statistics.fmean(confidences[index] for index in members)
            ece += (len(members) / len(labels)) * abs(bin_acc - bin_conf)
    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(true_prob),
    }, ids, labels


def verify_metrics(declared: object, expected: dict[str, float], label: str) -> None:
    require(isinstance(declared, dict), f"{label}: metrics missing")
    for key, value in expected.items():
        require(
            math.isclose(float(declared.get(key, float("nan"))), value, rel_tol=1e-6, abs_tol=1e-6),
            f"{label}: {key} mismatch",
        )


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(declared: object, expected: dict[str, float | int], label: str) -> None:
    require(isinstance(declared, dict), f"{label}: summary missing")
    require(int(declared.get("n", -1)) == int(expected["n"]), f"{label}: n mismatch")
    for key in ("mean", "std"):
        require(
            math.isclose(float(declared.get(key, float("nan"))), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: {key} mismatch",
        )


def paired_bootstrap_ci(deltas: Sequence[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    boot = [float(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))) for _ in range(samples)]
    boot.sort()
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def independently_verify_eligibility(
    source_path: Path,
    declared: object,
    *,
    required_choice_count: int,
) -> tuple[list, list]:
    require(isinstance(declared, dict), "eligibility block missing")
    source = load_arc_split(source_path)
    eligible = [example for example in source if len(example.choices) == required_choice_count]
    excluded = [example for example in source if len(example.choices) != required_choice_count]
    distribution = Counter(len(example.choices) for example in source)
    require(int(declared.get("source_rows", -1)) == len(source), "eligibility source row mismatch")
    require(declared.get("source_dataset_digest") == dataset_digest(source), "eligibility source dataset digest mismatch")
    require(declared.get("source_id_digest") == id_digest(source), "eligibility source ID digest mismatch")
    require(int(declared.get("required_choice_count", -1)) == required_choice_count, "eligibility choice count mismatch")
    require({int(k): int(v) for k, v in (declared.get("choice_count_distribution") or {}).items()} == dict(sorted(distribution.items())), "eligibility choice distribution mismatch")
    require(int(declared.get("eligible_rows", -1)) == len(eligible), "eligibility retained row mismatch")
    require(declared.get("eligible_dataset_digest") == dataset_digest(eligible), "eligibility retained dataset digest mismatch")
    require(declared.get("eligible_id_digest") == id_digest(eligible), "eligibility retained ID digest mismatch")
    require(int(declared.get("excluded_rows", -1)) == len(excluded), "eligibility excluded row mismatch")
    require(declared.get("excluded_id_digest") == id_digest(excluded), "eligibility excluded ID digest mismatch")
    expected_excluded = [{"id": example.item_id, "choice_count": len(example.choices)} for example in excluded]
    require(declared.get("excluded") == expected_excluded, "eligibility excluded-row evidence mismatch")
    used_rows = int(declared.get("used_rows", -1))
    require(0 < used_rows <= len(eligible), "invalid used-row count")
    used = eligible[:used_rows]
    require(declared.get("used_dataset_digest") == dataset_digest(used), "limit was not applied after eligibility: dataset digest mismatch")
    require(declared.get("used_id_digest") == id_digest(used), "limit was not applied after eligibility: ID digest mismatch")
    return eligible, used


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify eligible real-ARC protocol-v3 controls.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v3 is not frozen")
    frozen_eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(frozen_eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 required choice count drift")
    frozen_negative = frozen.get("negative_control") or {}
    frozen_ablations = frozen.get("ablations") or {}
    require(frozen_negative.get("type") == "deterministic training-label permutation", "negative-control type drift")
    require(frozen_negative.get("permutation_seed") == 20260807, "negative-control seed drift")
    require(set(frozen_ablations.get("required") or []) == {"no_planner", "no_target"}, "required ablations drift")

    protocol = payload.get("protocol") or {}
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol mismatch")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "wrong dataset")
    require(protocol.get("required_choice_count") == required_choice_count, "result choice-count contract mismatch")
    require(protocol.get("eligibility_rule") == frozen_eligibility.get("rule"), "result eligibility rule mismatch")
    train_eligible, train_used = independently_verify_eligibility(args.train, protocol.get("train_eligibility"), required_choice_count=required_choice_count)
    validation_eligible, validation_used = independently_verify_eligibility(args.validation, protocol.get("validation_eligibility"), required_choice_count=required_choice_count)
    require(protocol.get("eligible_train_validation_overlap") == 0, "declared eligible train/validation leakage")
    require(not ({e.item_id for e in train_eligible} & {e.item_id for e in validation_eligible}), "independent eligible train/validation leakage")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked test boundary weakened")
    require(protocol.get("negative_control_type") == frozen_negative["type"], "negative-control implementation drift")
    require(protocol.get("negative_control_seed") == frozen_negative["permutation_seed"], "negative-control seed mismatch")
    require(float(protocol.get("negative_control_max_validation_accuracy", -1.0)) == 0.35, "negative-control threshold drift")
    require(set(protocol.get("required_variants") or []) == {"full", "no_planner", "no_target"}, "variant set mismatch")
    require(protocol.get("original_training_label_digest") != protocol.get("permuted_training_label_digest"), "permuted labels did not change")
    claim = str(protocol.get("claim_boundary", ""))
    for phrase in ("five-seed/20-epoch", "locked-test", "mechanism", "superiority", "RESEARCH_COMPLETE"):
        require(phrase in claim, f"claim boundary missing: {phrase}")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid development seeds")
    model_steps = int(protocol.get("model_steps", 0))
    require(model_steps >= 1, "planner must be exercised")
    canonical_ids = [example.item_id for example in validation_used]
    canonical_labels = [example.label for example in validation_used]

    variants = payload.get("variants")
    require(isinstance(variants, dict) and set(variants) == {"full", "no_planner", "no_target"}, "variant payload mismatch")
    accuracies: dict[str, list[float]] = {}
    for variant in ("full", "no_planner", "no_target"):
        block = variants[variant]
        records = block.get("records")
        require(isinstance(records, list) and len(records) == len(seeds), f"{variant}: seed records missing")
        values: list[float] = []
        for expected_seed, record in zip(seeds, records, strict=True):
            require(record.get("seed") == expected_seed, f"{variant}: seed mismatch")
            expected_use_planner = variant != "no_planner"
            expected_use_target = variant != "no_target"
            require(record.get("use_planner") is expected_use_planner, f"{variant}: planner flag mismatch")
            require(record.get("use_target") is expected_use_target, f"{variant}: target flag mismatch")
            expected_actions = model_steps if expected_use_planner else 0
            require(record.get("expected_action_steps") == expected_actions, f"{variant}: expected action count mismatch")
            for key in ("observed_action_steps", "observed_reversed_action_steps"):
                observed = record.get(key)
                require(isinstance(observed, list) and observed and all(step == expected_actions for step in observed), f"{variant}: {key} mismatch")
            recomputed, ids, labels = metrics_from_rows(record.get("predictions"))
            rev_metrics, rev_ids, rev_labels = metrics_from_rows(record.get("choice_reversal_predictions"))
            require(ids == canonical_ids and labels == canonical_labels, f"{variant}: eligible validation pairing mismatch")
            require(rev_ids == canonical_ids, f"{variant}: choice reversal changed item identity")
            require(rev_labels == [3 - label for label in canonical_labels], f"{variant}: choice-reversal label remapping failed")
            verify_metrics(record.get("metrics"), recomputed, f"{variant}/seed={expected_seed}")
            verify_metrics(record.get("choice_reversal_metrics"), rev_metrics, f"{variant}/seed={expected_seed}/reversed")
            values.append(recomputed["accuracy"])
        accuracies[variant] = values
        verify_summary(block.get("accuracy"), summarize(values), f"{variant}/accuracy")

    paired = payload.get("paired_effects")
    require(isinstance(paired, dict) and set(paired) == {"no_planner", "no_target"}, "paired effects missing")
    for offset, variant in enumerate(("no_planner", "no_target")):
        deltas = [full - ablated for full, ablated in zip(accuracies["full"], accuracies[variant], strict=True)]
        block = paired[variant]
        require(block.get("seed_level_full_minus_ablation") == deltas, f"{variant}: paired delta mismatch")
        mean_delta = float(statistics.fmean(deltas))
        std_delta = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
        ci_low, ci_high = paired_bootstrap_ci(deltas, seed=20260807 + offset)
        require(math.isclose(float(block.get("mean_full_minus_ablation")), mean_delta, rel_tol=1e-12, abs_tol=1e-12), f"{variant}: mean delta mismatch")
        require(math.isclose(float(block.get("std_paired_difference")), std_delta, rel_tol=1e-12, abs_tol=1e-12), f"{variant}: paired std mismatch")
        require(math.isclose(float(block.get("paired_bootstrap_ci95_low")), ci_low, rel_tol=1e-12, abs_tol=1e-12), f"{variant}: CI low mismatch")
        require(math.isclose(float(block.get("paired_bootstrap_ci95_high")), ci_high, rel_tol=1e-12, abs_tol=1e-12), f"{variant}: CI high mismatch")
        expected_numeric = mean_delta >= 0.01 and ci_low > 0.0
        require(block.get("observed_mechanism_numeric_criterion_met") is expected_numeric, f"{variant}: numeric criterion flag mismatch")

    negative = payload.get("negative_control")
    require(isinstance(negative, dict), "negative-control payload missing")
    records = negative.get("records")
    require(isinstance(records, list) and len(records) == len(seeds), "negative-control seed records missing")
    negative_values: list[float] = []
    for expected_seed, record in zip(seeds, records, strict=True):
        require(record.get("seed") == expected_seed, "negative-control seed mismatch")
        recomputed, ids, labels = metrics_from_rows(record.get("predictions"))
        require(ids == canonical_ids and labels == canonical_labels, "negative control changed eligible validation rows/labels")
        observed = record.get("observed_action_steps")
        require(isinstance(observed, list) and observed and all(step == model_steps for step in observed), "negative control did not exercise planner")
        verify_metrics(record.get("metrics"), recomputed, f"negative/seed={expected_seed}")
        negative_values.append(recomputed["accuracy"])
    negative_summary = summarize(negative_values)
    verify_summary(negative.get("accuracy"), negative_summary, "negative-control/accuracy")
    negative_pass = float(negative_summary["mean"]) <= 0.35
    require(negative.get("pass") is negative_pass, "negative-control pass flag mismatch")
    require(negative_pass, f"negative control failed: mean accuracy {negative_summary['mean']:.6f} > 0.35")

    report = {
        "verdict": "PROTOCOL_V3_ARC_CONTROLS_EXECUTION_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "train_source_rows": len(load_arc_split(args.train)),
        "train_eligible_rows": len(train_eligible),
        "train_used_rows": len(train_used),
        "validation_source_rows": len(load_arc_split(args.validation)),
        "validation_eligible_rows": len(validation_eligible),
        "validation_used_rows": len(validation_used),
        "seeds": seeds,
        "variant_accuracy": {variant: summarize(values) for variant, values in accuracies.items()},
        "paired_effects": paired,
        "negative_control_accuracy": negative_summary,
        "negative_control_pass": True,
        "locked_test_evaluated": False,
        "final_five_seed_20_epoch_protocol_executed": False,
        "mechanism_claim_authorized": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
