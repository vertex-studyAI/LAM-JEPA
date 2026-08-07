from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

EXPECTED_PROTOCOL_ID = "lam-jepa-arc-challenge-v1"
EXPECTED_MODEL_ID = "microsoft/deberta-v3-xsmall"
EXPECTED_MODEL_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
EXPECTED_LICENSE = "MIT"
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_SENTENCEPIECE_VERSION = "0.2.2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected object JSON: {path}")
    return payload


def recompute_metrics(rows: object, expected_ids: list[str] | None, name: str):
    require(isinstance(rows, list) and rows, f"{name}: predictions must be a non-empty list")
    ids: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    for row in rows:
        require(isinstance(row, dict), f"{name}: row must be an object")
        item_id = str(row.get("id", ""))
        label = row.get("label")
        prediction = row.get("prediction")
        values = row.get("probabilities")
        require(item_id, f"{name}: item id missing")
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
        require(ids == expected_ids, f"{name}: item identity/order changed")

    n = len(ids)
    accuracy = sum(int(prediction == label) for prediction, label in zip(predictions, labels, strict=True)) / n
    brier = sum(
        sum((value - (1.0 if index == label else 0.0)) ** 2 for index, value in enumerate(values))
        for values, label in zip(probabilities, labels, strict=True)
    ) / n
    true_probability = statistics.fmean(values[label] for values, label in zip(probabilities, labels, strict=True))
    confidences = [max(values) for values in probabilities]
    correct = [float(prediction == label) for prediction, label in zip(predictions, labels, strict=True)]
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
            ece += (len(members) / n) * abs(
                statistics.fmean(correct[index] for index in members)
                - statistics.fmean(confidences[index] for index in members)
            )
    return {
        "accuracy": float(accuracy),
        "brier": float(brier),
        "ece": float(ece),
        "mean_true_class_probability": float(true_probability),
    }, ids, labels


def verify_metric_row(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: metrics missing")
    for key, expected_value in expected.items():
        require(key in actual, f"{name}: missing metric {key}")
        require(
            math.isclose(float(actual[key]), expected_value, rel_tol=1e-6, abs_tol=1e-6),
            f"{name}: metric mismatch for {key}",
        )


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(actual: object, expected: dict[str, float | int], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: summary missing")
    require(actual.get("n") == expected["n"], f"{name}: n mismatch")
    for key in ("mean", "std"):
        require(
            math.isclose(float(actual[key]), float(expected[key]), rel_tol=1e-9, abs_tol=1e-9),
            f"{name}: {key} mismatch",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen protocol-v1 DeBERTa ARC smoke evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.results)
    frozen = read_json(args.protocol)
    require(payload.get("artifact_type") == "lam-jepa frozen ARC protocol-v1 DeBERTa development smoke", "artifact type changed")
    protocol = payload.get("protocol")
    records = payload.get("records")
    aggregate = payload.get("summary")
    require(isinstance(protocol, dict), "artifact protocol missing")
    require(isinstance(records, list), "artifact records missing")
    require(isinstance(aggregate, dict), "artifact summary missing")

    require(frozen.get("protocol_id") == EXPECTED_PROTOCOL_ID, "frozen protocol id changed")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "frozen protocol status changed")
    frozen_model = frozen.get("models", {}).get("strong_pretrained_baseline", {})
    require(frozen_model.get("model") == EXPECTED_MODEL_ID, "frozen model id changed")
    require(frozen_model.get("revision") == EXPECTED_MODEL_REVISION, "frozen model revision changed")
    require(frozen_model.get("license") == EXPECTED_LICENSE, "frozen model license changed")
    require(frozen.get("training_budget", {}).get("training_seeds") == [1, 2, 3, 4, 5], "frozen seeds changed")
    require(frozen.get("training_budget", {}).get("epochs") == 20, "frozen epochs changed")
    require(frozen.get("training_budget", {}).get("batch_size") == 32, "frozen batch size changed")
    require(float(frozen.get("training_budget", {}).get("pretrained_baseline_learning_rate")) == 2e-5, "frozen pretrained LR changed")

    require(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "artifact protocol id mismatch")
    require(protocol.get("protocol_sha256") == sha256_file(args.protocol), "artifact protocol hash mismatch")
    require(protocol.get("protocol_status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "artifact protocol status mismatch")
    require(protocol.get("development_smoke_only") is True, "development-smoke boundary missing")
    require(protocol.get("confirmatory_budget_executed") is False, "artifact falsely claims confirmatory budget")
    require(protocol.get("test_split_accessed") is False, "artifact accessed test split")
    require(protocol.get("frozen_confirmatory_seeds") == [1, 2, 3, 4, 5], "artifact frozen seed record changed")
    require(protocol.get("frozen_confirmatory_epochs") == 20, "artifact frozen epoch record changed")
    require(protocol.get("frozen_confirmatory_batch_size") == 32, "artifact frozen batch-size record changed")
    require(float(protocol.get("pretrained_learning_rate")) == 2e-5, "artifact pretrained LR drift")
    require(float(protocol.get("lam_jepa_learning_rate")) == 3e-4, "artifact LAM LR drift")
    require(protocol.get("pretrained_model_id") == EXPECTED_MODEL_ID, "artifact model id mismatch")
    require(protocol.get("pretrained_model_revision") == EXPECTED_MODEL_REVISION, "artifact model revision mismatch")
    require(protocol.get("resolved_pretrained_revision") == EXPECTED_MODEL_REVISION, "remote model revision mismatch")
    require(protocol.get("pretrained_model_license") == EXPECTED_LICENSE, "artifact license mismatch")
    require(protocol.get("pretrained_weight_file") == "pytorch_model.bin", "unexpected frozen weight filename")
    require(protocol.get("pretrained_weight_format") == "pytorch_pickle_bin", "weight format must remain explicit")
    require(protocol.get("trust_remote_code") is False, "remote code must stay disabled")
    require(protocol.get("transformers_version") == EXPECTED_TRANSFORMERS_VERSION, "transformers runtime drift")
    require(protocol.get("sentencepiece_version") == EXPECTED_SENTENCEPIECE_VERSION, "sentencepiece runtime drift")
    require("not parameter matched" in str(protocol.get("parameter_matching", "")), "parameter mismatch boundary missing")

    total_parameters = int(protocol.get("pretrained_total_parameters", 0))
    trainable_parameters = int(protocol.get("pretrained_trainable_parameters", 0))
    require(total_parameters > 10_000_000, "DeBERTa total parameter count implausibly small")
    require(0 < trainable_parameters <= total_parameters, "invalid DeBERTa trainable parameter count")

    seeds = protocol.get("seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2 and len(seeds) == len(set(seeds)), "invalid development seed set")
    require(len(records) == len(seeds), "record count does not match development seed count")
    require(protocol.get("train_validation_overlap") == 0, "train/validation leakage detected")
    for key in ("train_digest", "validation_digest", "train_id_digest", "validation_id_digest"):
        digest = protocol.get(key)
        require(isinstance(digest, str) and len(digest) == 64, f"invalid digest: {key}")
    require(protocol.get("train_digest") != protocol.get("validation_digest"), "train/validation dataset digests collide")
    require(protocol.get("train_id_digest") != protocol.get("validation_id_digest"), "train/validation ID digests collide")

    claim = str(protocol.get("claim_boundary", ""))
    for phrase in ("five-seed", "20-epoch", "ARC test", "not compute matched", "independent reproduction", "RESEARCH_COMPLETE"):
        require(phrase in claim, f"claim boundary missing phrase: {phrase}")

    validation_n = int(protocol.get("validation_examples", 0))
    require(validation_n > 0, "validation rows missing")
    canonical_ids: list[str] | None = None
    canonical_labels: list[int] | None = None
    lam_accuracy: list[float] = []
    deberta_accuracy: list[float] = []
    deltas: list[float] = []

    for expected_seed, record in zip(seeds, records, strict=True):
        require(isinstance(record, dict) and record.get("seed") == expected_seed, "seed record mismatch")
        deberta = record.get("frozen_deberta")
        lam = record.get("lam_jepa")
        require(isinstance(deberta, dict) and isinstance(lam, dict), "model record missing")
        require(int(deberta.get("training_steps_executed", 0)) >= 1, "DeBERTa executed zero training steps")
        for label, row in (("deberta", deberta), ("lam", lam)):
            for field in ("training_wall_seconds", "validation_wall_seconds", "choice_reversal_wall_seconds"):
                value = float(row.get(field, 0.0))
                require(math.isfinite(value) and value > 0.0, f"{label}: invalid compute evidence {field}")

        deberta_metrics, ids, labels = recompute_metrics(deberta.get("predictions"), canonical_ids, f"seed {expected_seed}/deberta")
        if canonical_ids is None:
            canonical_ids = ids
            canonical_labels = labels
        else:
            require(labels == canonical_labels, f"seed {expected_seed}: validation labels changed")
        require(len(ids) == validation_n, f"seed {expected_seed}: validation count mismatch")

        lam_metrics, lam_ids, lam_labels = recompute_metrics(lam.get("predictions"), canonical_ids, f"seed {expected_seed}/lam")
        require(lam_ids == canonical_ids and lam_labels == labels, f"seed {expected_seed}: cross-model row/label mismatch")

        deberta_rev_metrics, deberta_rev_ids, deberta_rev_labels = recompute_metrics(
            deberta.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/deberta-reversed"
        )
        lam_rev_metrics, lam_rev_ids, lam_rev_labels = recompute_metrics(
            lam.get("choice_reversal_predictions"), canonical_ids, f"seed {expected_seed}/lam-reversed"
        )
        require(deberta_rev_ids == canonical_ids and lam_rev_ids == canonical_ids, "choice reversal changed item identity")
        require(deberta_rev_labels == lam_rev_labels, "choice reversal labels differ across models")
        require(deberta_rev_labels == [3 - value for value in labels], "choice reversal label remapping invalid")

        verify_metric_row(deberta.get("metrics"), deberta_metrics, f"seed {expected_seed}/deberta")
        verify_metric_row(lam.get("metrics"), lam_metrics, f"seed {expected_seed}/lam")
        verify_metric_row(deberta.get("choice_reversal_metrics"), deberta_rev_metrics, f"seed {expected_seed}/deberta-reversed")
        verify_metric_row(lam.get("choice_reversal_metrics"), lam_rev_metrics, f"seed {expected_seed}/lam-reversed")

        delta = float(lam_metrics["accuracy"] - deberta_metrics["accuracy"])
        require(
            math.isclose(float(record.get("accuracy_delta_lam_minus_deberta")), delta, rel_tol=1e-9, abs_tol=1e-9),
            f"seed {expected_seed}: paired delta mismatch",
        )
        lam_accuracy.append(lam_metrics["accuracy"])
        deberta_accuracy.append(deberta_metrics["accuracy"])
        deltas.append(delta)

    verify_summary(aggregate.get("lam_accuracy"), summarize(lam_accuracy), "lam_accuracy")
    verify_summary(aggregate.get("deberta_accuracy"), summarize(deberta_accuracy), "deberta_accuracy")
    verify_summary(aggregate.get("paired_accuracy_delta_lam_minus_deberta"), summarize(deltas), "paired_delta")

    report = {
        "verdict": "FROZEN_DEBERTA_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "development_seeds": seeds,
        "confirmatory_budget_executed": False,
        "test_split_accessed": False,
        "claim_boundary_preserved": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
