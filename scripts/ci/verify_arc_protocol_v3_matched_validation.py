from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

BOOTSTRAP_SEED = 20260807
BOOTSTRAP_DRAWS = 20000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected object JSON: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_rows(rows: object, expected_ids: list[str] | None, name: str):
    require(isinstance(rows, list) and rows, f"{name}: predictions missing")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    for row in rows:
        require(isinstance(row, dict), f"{name}: row must be object")
        item_id = str(row.get("id", ""))
        label = row.get("label")
        prediction = row.get("prediction")
        values = row.get("probabilities")
        require(item_id, f"{name}: missing id")
        require(isinstance(label, int) and 0 <= label < 4, f"{name}/{item_id}: invalid label")
        require(isinstance(prediction, int) and 0 <= prediction < 4, f"{name}/{item_id}: invalid prediction")
        require(isinstance(values, list) and len(values) == 4, f"{name}/{item_id}: expected four probabilities")
        values = [float(value) for value in values]
        require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), f"{name}/{item_id}: invalid probability")
        require(math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5), f"{name}/{item_id}: probabilities do not sum to one")
        require(prediction == max(range(4), key=values.__getitem__), f"{name}/{item_id}: prediction is not argmax")
        ids.append(item_id)
        labels.append(label)
        predictions.append(prediction)
        probabilities.append(values)
    require(len(ids) == len(set(ids)), f"{name}: duplicate ids")
    if expected_ids is not None:
        require(ids == expected_ids, f"{name}: row identity/order mismatch")

    n = len(ids)
    correct = [float(pred == label) for pred, label in zip(predictions, labels, strict=True)]
    accuracy = statistics.fmean(correct)
    brier = statistics.fmean(
        sum((value - (1.0 if index == label else 0.0)) ** 2 for index, value in enumerate(values))
        for values, label in zip(probabilities, labels, strict=True)
    )
    mean_true = statistics.fmean(values[label] for values, label in zip(probabilities, labels, strict=True))
    confidences = [max(values) for values in probabilities]
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            i for i, confidence in enumerate(confidences)
            if (confidence >= lower if bin_index == 0 else confidence > lower) and confidence <= upper
        ]
        if members:
            ece += (len(members) / n) * abs(
                statistics.fmean(correct[i] for i in members)
                - statistics.fmean(confidences[i] for i in members)
            )
    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(mean_true),
    }, ids, labels, predictions, probabilities


def verify_metrics(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: metrics missing")
    for key, expected_value in expected.items():
        require(key in actual, f"{name}: missing metric {key}")
        require(math.isclose(float(actual[key]), expected_value, rel_tol=1e-6, abs_tol=1e-6), f"{name}: {key} mismatch")


def bootstrap_ci(values: list[float]) -> list[float]:
    require(bool(values), "cannot bootstrap empty values")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    lower = max(0, int(math.floor(0.025 * (BOOTSTRAP_DRAWS - 1))))
    upper = min(BOOTSTRAP_DRAWS - 1, int(math.ceil(0.975 * (BOOTSTRAP_DRAWS - 1))))
    return [float(means[lower]), float(means[upper])]


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "ci95": bootstrap_ci(values),
    }


def verify_summary(actual: object, expected: dict[str, object], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: summary missing")
    require(actual.get("n") == expected["n"], f"{name}: n mismatch")
    for key in ("mean", "std"):
        require(math.isclose(float(actual[key]), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9), f"{name}: {key} mismatch")
    actual_ci = actual.get("ci95")
    require(isinstance(actual_ci, list) and len(actual_ci) == 2, f"{name}: ci95 missing")
    for a, e in zip(actual_ci, expected["ci95"], strict=True):
        require(math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-9), f"{name}: ci95 mismatch")


def diagnostics(predictions: list[int], probabilities: list[list[float]]) -> dict[str, object]:
    ranges = [
        max(row[index] for row in probabilities) - min(row[index] for row in probabilities)
        for index in range(4)
    ]
    return {
        "prediction_class_count": len(set(predictions)),
        "prediction_histogram": {str(index): predictions.count(index) for index in range(4)},
        "unique_probability_rows_6dp": len({tuple(round(value, 6) for value in row) for row in probabilities}),
        "per_class_probability_ranges": ranges,
        "maximum_probability_range": max(ranges),
    }


def verify_diagnostics(actual: object, expected: dict[str, object], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: diagnostics missing")
    for key in ("prediction_class_count", "prediction_histogram", "unique_probability_rows_6dp"):
        require(actual.get(key) == expected[key], f"{name}: diagnostic mismatch {key}")
    for key in ("per_class_probability_ranges",):
        av = actual.get(key)
        ev = expected[key]
        require(isinstance(av, list) and len(av) == len(ev), f"{name}: {key} malformed")
        for a, e in zip(av, ev, strict=True):
            require(math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-9), f"{name}: {key} mismatch")
    require(math.isclose(float(actual.get("maximum_probability_range")), float(expected["maximum_probability_range"]), rel_tol=1e-9, abs_tol=1e-9), f"{name}: maximum probability range mismatch")


def verify_eligibility(summary: object, *, source: int, eligible: int, excluded_ids: set[str], name: str) -> None:
    require(isinstance(summary, dict), f"{name}: eligibility missing")
    require(summary.get("source_count") == source, f"{name}: source count mismatch")
    require(summary.get("eligible_count") == eligible, f"{name}: eligible count mismatch")
    require(summary.get("excluded_count") == source - eligible, f"{name}: excluded count mismatch")
    excluded = summary.get("excluded")
    require(isinstance(excluded, list), f"{name}: excluded rows missing")
    actual_ids = {str(row.get("id")) for row in excluded}
    require(actual_ids == excluded_ids, f"{name}: excluded IDs changed: {actual_ids}")
    require(all(int(row.get("choice_count", 4)) != 4 for row in excluded), f"{name}: eligible four-choice row was excluded")
    require(isinstance(summary.get("eligible_id_digest"), str) and len(summary["eligible_id_digest"]) == 64, f"{name}: eligible digest invalid")
    require(isinstance(summary.get("excluded_id_digest"), str) and len(summary["excluded_id_digest"]) == 64, f"{name}: excluded digest invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full protocol-v3 matched-capacity ARC validation evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.results)
    frozen = read_json(args.protocol)
    require(payload.get("artifact_type") == "lam-jepa ARC protocol-v3 full matched-capacity validation package", "artifact type changed")
    protocol = payload.get("protocol")
    records = payload.get("records")
    aggregate = payload.get("summary")
    require(isinstance(protocol, dict) and isinstance(records, list) and isinstance(aggregate, dict), "artifact structure invalid")

    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "frozen protocol id changed")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol no longer frozen")
    require(protocol.get("protocol_id") == frozen["protocol_id"], "artifact protocol id mismatch")
    require(protocol.get("protocol_sha256") == sha256_file(args.protocol), "protocol hash mismatch")
    require(protocol.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "artifact freeze status mismatch")
    require(protocol.get("confirmatory_test_accessed") is False, "confirmatory test was accessed")
    require(protocol.get("eligibility_rule") == "len(choices) == 4", "eligibility rule changed")
    require(protocol.get("training_seeds") == [1, 2, 3, 4, 5], "five-seed set changed")
    require(protocol.get("epochs") == 20 and protocol.get("batch_size") == 32, "full frozen training budget not executed")
    require(float(protocol.get("lam_jepa_learning_rate")) == 3e-4, "LAM learning rate changed")
    require(float(protocol.get("matched_baseline_learning_rate")) == 3e-4, "matched learning rate changed")
    require(protocol.get("model_steps") == 1, "model_steps changed")
    require(protocol.get("train_examples_used") == 1117, "not all eligible train rows used")
    require(protocol.get("validation_examples_used") == 295, "not all eligible validation rows used")

    verify_eligibility(
        protocol.get("train_eligibility"), source=1119, eligible=1117,
        excluded_ids={"NYSEDREGENTS_2004_4_8", "TIMSS_1995_8_N3"}, name="train",
    )
    verify_eligibility(
        protocol.get("validation_eligibility"), source=299, eligible=295,
        excluded_ids={"NYSEDREGENTS_2014_4_4", "NYSEDREGENTS_2014_4_19", "NYSEDREGENTS_2014_4_28", "TIMSS_2003_8_pg29"}, name="validation",
    )

    lam_active = int(protocol.get("lam_gradient_active_parameters", 0))
    lam_total = int(protocol.get("lam_total_trainable_parameters", 0))
    matched_total = int(protocol.get("matched_supervised_trainable_parameters", 0))
    matched_active = int(protocol.get("matched_supervised_gradient_active_parameters", 0))
    require(0 < lam_active <= lam_total, "invalid LAM active/total parameter counts")
    require(matched_total == matched_active > 0, "matched baseline includes inactive trainable parameters")
    ratio = matched_total / lam_active
    require(0.99 <= ratio <= 1.01, f"matched parameter ratio violates frozen v3 gate: {ratio}")
    require(math.isclose(float(protocol.get("parameter_ratio_matched_to_lam_active")), ratio, rel_tol=1e-12, abs_tol=1e-12), "stored parameter ratio mismatch")
    require(math.isclose(float(protocol.get("parameter_relative_gap")), abs(matched_total-lam_active)/lam_active, rel_tol=1e-12, abs_tol=1e-12), "stored parameter gap mismatch")

    require(len(records) == 5, "expected five seed records")
    canonical_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    lam_accuracy: list[float] = []
    matched_accuracy: list[float] = []
    deltas: list[float] = []
    lam_drops: list[float] = []
    matched_drops: list[float] = []

    for expected_seed, record in zip([1,2,3,4,5], records, strict=True):
        require(isinstance(record, dict) and record.get("seed") == expected_seed, "seed record mismatch")
        lam = record.get("lam_jepa")
        matched = record.get("matched_supervised")
        require(isinstance(lam, dict) and isinstance(matched, dict), "model record missing")
        for model_name, model_record in (("lam",lam),("matched",matched)):
            for timing_key in ("training_wall_seconds","validation_wall_seconds"):
                value=float(model_record.get(timing_key,0.0))
                require(math.isfinite(value) and value>0.0, f"{model_name}: invalid timing evidence {timing_key}")

        lam_metrics, ids, labels, predictions, probabilities = metrics_from_rows(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        if canonical_ids is None:
            canonical_ids=ids
            canonical_labels=labels
        else:
            require(labels==canonical_labels, f"seed {expected_seed}: LAM labels changed")
        matched_metrics, matched_ids, matched_labels, matched_predictions, matched_probabilities = metrics_from_rows(matched.get("predictions"), canonical_ids, f"seed {expected_seed}/matched")
        require(matched_ids==canonical_ids and matched_labels==canonical_labels, f"seed {expected_seed}: cross-model rows/labels differ")
        require(len(ids)==295, f"seed {expected_seed}: missing eligible validation rows")

        verify_metrics(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metrics(matched.get("metrics"), matched_metrics, f"seed {expected_seed}/matched")
        verify_diagnostics(lam.get("diagnostics"), diagnostics(predictions,probabilities), f"seed {expected_seed}/lam")
        verify_diagnostics(matched.get("diagnostics"), diagnostics(matched_predictions,matched_probabilities), f"seed {expected_seed}/matched")

        lam_rev_metrics, rev_ids, rev_labels, _, _ = metrics_from_rows(lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed")
        matched_rev_metrics, matched_rev_ids, matched_rev_labels, _, _ = metrics_from_rows(matched.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/matched-reversed")
        require(rev_ids==canonical_ids and matched_rev_ids==canonical_ids, "reversal changed row identity")
        require(rev_labels==matched_rev_labels==[3-label for label in canonical_labels], "reversal label remap invalid")
        verify_metrics(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {expected_seed}/lam-reversed")
        verify_metrics(matched.get("choice_reversal_metrics"), matched_rev_metrics, f"seed {expected_seed}/matched-reversed")

        delta=lam_metrics["accuracy"]-matched_metrics["accuracy"]
        lam_drop=lam_metrics["accuracy"]-lam_rev_metrics["accuracy"]
        matched_drop=matched_metrics["accuracy"]-matched_rev_metrics["accuracy"]
        require(math.isclose(float(record.get("accuracy_delta_lam_minus_matched")),delta,rel_tol=1e-9,abs_tol=1e-9),"paired delta mismatch")
        require(math.isclose(float(lam.get("robustness_accuracy_drop")),lam_drop,rel_tol=1e-9,abs_tol=1e-9),"LAM robustness drop mismatch")
        require(math.isclose(float(matched.get("robustness_accuracy_drop")),matched_drop,rel_tol=1e-9,abs_tol=1e-9),"matched robustness drop mismatch")
        lam_accuracy.append(lam_metrics["accuracy"])
        matched_accuracy.append(matched_metrics["accuracy"])
        deltas.append(delta)
        lam_drops.append(lam_drop)
        matched_drops.append(matched_drop)

    require(canonical_ids is not None and len(canonical_ids)==295, "canonical validation identity incomplete")
    require(len(set(canonical_ids))==295, "duplicate eligible validation IDs")
    verify_summary(aggregate.get("lam_accuracy"), summarize(lam_accuracy), "lam_accuracy")
    verify_summary(aggregate.get("matched_supervised_accuracy"), summarize(matched_accuracy), "matched_accuracy")
    paired=summarize(deltas)
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_matched"),paired,"paired_delta")
    lam_robust=summarize(lam_drops)
    matched_robust=summarize(matched_drops)
    verify_summary(aggregate.get("lam_choice_reversal_accuracy_drop"),lam_robust,"lam_robustness")
    verify_summary(aggregate.get("matched_choice_reversal_accuracy_drop"),matched_robust,"matched_robustness")

    effect=float(frozen["metrics"]["practical_effect_threshold_absolute"])
    expected_superiority=float(paired["mean"])>=effect and float(paired["ci95"][0])>0.0
    require(bool(aggregate.get("matched_superiority_gate"))==expected_superiority,"matched superiority gate mismatch")
    limit=float(frozen["robustness"]["maximum_allowed_lam_accuracy_drop"])
    expected_robust=float(lam_robust["mean"])<=limit and float(lam_robust["mean"])-float(matched_robust["mean"])<=0.02
    require(bool(aggregate.get("robustness_gate_against_matched"))==expected_robust,"robustness gate mismatch")

    report={
        "verdict":"PROTOCOL_V3_MATCHED_FULL_VALIDATION_VERIFIED_ONLY",
        "protocol_id":protocol["protocol_id"],
        "seeds":[1,2,3,4,5],
        "epochs":20,
        "train_eligible":1117,
        "validation_eligible":295,
        "parameter_ratio":ratio,
        "lam_mean_accuracy":lam_accuracy and statistics.fmean(lam_accuracy),
        "matched_mean_accuracy":matched_accuracy and statistics.fmean(matched_accuracy),
        "paired_mean_delta":statistics.fmean(deltas),
        "paired_ci95":paired["ci95"],
        "matched_superiority_gate":expected_superiority,
        "robustness_gate_against_matched":expected_robust,
        "confirmatory_test_accessed":False,
        "research_complete":False,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
