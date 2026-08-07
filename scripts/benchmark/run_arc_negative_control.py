from __future__ import annotations

import argparse
import hashlib
import json
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

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    _predict_lam,
    _train_lam_jepa,
    dataset_digest,
    id_digest,
    load_arc_split,
    score_predictions,
)
from lam_jepa.model import LAMJEPAConfig

EXPECTED_PROTOCOL_ID = "lam-jepa-arc-challenge-v2"
EXPECTED_PERMUTATION_SEED = 20260807
EXPECTED_FAILURE_THRESHOLD = 0.35


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"protocol not found: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise RuntimeError(f"expected frozen protocol {EXPECTED_PROTOCOL_ID}")
    if protocol.get("status") != "FROZEN_BEFORE_CONFIRMATORY_TEST":
        raise RuntimeError("protocol v2 is not frozen")
    negative = protocol.get("negative_control", {})
    if negative.get("type") != "deterministic training-label permutation":
        raise RuntimeError("negative-control type drifted")
    if negative.get("permutation_seed") != EXPECTED_PERMUTATION_SEED:
        raise RuntimeError("negative-control permutation seed drifted")
    if negative.get("split") != "validation only; never use confirmatory test labels for the negative control":
        raise RuntimeError("negative-control split policy drifted")
    if "0.35" not in str(negative.get("failure_rule", "")):
        raise RuntimeError("negative-control failure threshold drifted")
    return protocol, sha256_file(path)


def permute_training_labels(examples: Sequence[ARCExample], seed: int) -> tuple[list[ARCExample], dict]:
    if len(examples) < 4:
        raise ValueError("negative-control permutation requires at least four training examples")
    original_labels = [example.label for example in examples]
    permuted_labels = list(original_labels)
    random.Random(seed).shuffle(permuted_labels)
    changed = sum(int(before != after) for before, after in zip(original_labels, permuted_labels, strict=True))
    if changed == 0:
        raise RuntimeError("deterministic label permutation changed zero labels")
    if Counter(original_labels) != Counter(permuted_labels):
        raise RuntimeError("label permutation changed the training-label multiset")

    permuted = [
        ARCExample(
            item_id=example.item_id,
            question=example.question,
            choices=example.choices,
            label=permuted_label,
        )
        for example, permuted_label in zip(examples, permuted_labels, strict=True)
    ]
    mapping_digest = hashlib.sha256()
    mapping_rows: list[dict[str, int | str]] = []
    for example, before, after in zip(examples, original_labels, permuted_labels, strict=True):
        row = {"id": example.item_id, "original_label": before, "permuted_label": after}
        mapping_rows.append(row)
        mapping_digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        mapping_digest.update(b"\n")

    return permuted, {
        "permutation_seed": seed,
        "changed_label_count": changed,
        "unchanged_label_count": len(examples) - changed,
        "original_label_counts": dict(sorted(Counter(original_labels).items())),
        "permuted_label_counts": dict(sorted(Counter(permuted_labels).items())),
        "mapping_digest": mapping_digest.hexdigest(),
        "mapping": mapping_rows,
    }


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen ARC protocol-v2 shuffled-label negative control.")
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.model_steps) < 1:
        parser.error("epochs, batch size, and model steps must be positive")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        parser.error("development negative-control smoke requires at least two unique training seeds")

    protocol, protocol_sha256 = load_protocol(args.protocol)
    if args.learning_rate != float(protocol["training_budget"]["lam_jepa_learning_rate"]):
        parser.error("LAM learning rate must match frozen protocol v2")
    if args.model_steps != int(protocol["training_budget"]["model_steps"]):
        parser.error("LAM model_steps must match frozen protocol v2")

    train_all = load_arc_split(args.train)
    validation_all = load_arc_split(args.validation)
    train = list(train_all[: args.train_limit] if args.train_limit else train_all)
    validation = list(validation_all[: args.validation_limit] if args.validation_limit else validation_all)
    if not train or not validation:
        parser.error("train and validation splits must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        parser.error("current ARC protocol requires exactly four choices")
    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    if overlap:
        raise SystemExit(f"train/validation leakage detected: {overlap[:5]}")

    permuted_train, permutation = permute_training_labels(train, EXPECTED_PERMUTATION_SEED)
    cfg = LAMJEPAConfig()
    records: list[dict] = []
    accuracies: list[float] = []

    for seed in seeds:
        model = _train_lam_jepa(
            permuted_train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_steps=args.model_steps,
            device=args.device,
        )
        probabilities, labels, rows = _predict_lam(
            model,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        metrics = score_predictions(probabilities, labels)
        accuracy = float(metrics["accuracy"])
        accuracies.append(accuracy)
        records.append(
            {
                "training_seed": seed,
                "metrics": metrics,
                "threshold_exceeded": accuracy > EXPECTED_FAILURE_THRESHOLD,
                "predictions": rows,
            }
        )

    threshold_exceeded_seeds = [
        int(record["training_seed"])
        for record in records
        if bool(record["threshold_exceeded"])
    ]
    payload = {
        "artifact_type": "lam-jepa ARC protocol-v2 shuffled-label negative-control development smoke",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "status": protocol["status"],
            "development_smoke_only": True,
            "confirmatory_test_accessed": False,
            "negative_control_type": protocol["negative_control"]["type"],
            "permutation_seed": EXPECTED_PERMUTATION_SEED,
            "failure_threshold_accuracy": EXPECTED_FAILURE_THRESHOLD,
            "failure_rule": protocol["negative_control"]["failure_rule"],
            "training_seeds": seeds,
            "smoke_epochs": args.epochs,
            "smoke_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_steps": args.model_steps,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "original_train_digest": dataset_digest(train),
            "permuted_train_digest": dataset_digest(permuted_train),
            "train_id_digest": id_digest(train),
            "permuted_train_id_digest": id_digest(permuted_train),
            "validation_digest": dataset_digest(validation),
            "validation_id_digest": id_digest(validation),
            "permutation": permutation,
            "claim_boundary": (
                "This is a validation-only development smoke for the preregistered shuffled-label negative control. "
                "It does not access ARC test, does not execute the five-seed 20-epoch confirmatory budget, and does not establish research completion."
            ),
        },
        "records": records,
        "summary": {
            "validation_accuracy": summarize(accuracies),
            "threshold_exceeded_seeds": threshold_exceeded_seeds,
            "failure_condition_triggered": bool(threshold_exceeded_seeds),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))

    if threshold_exceeded_seeds:
        raise SystemExit(
            "negative-control leakage/shortcut stop condition triggered for training seeds: "
            + ", ".join(map(str, threshold_exceeded_seeds))
        )


if __name__ == "__main__":
    main()
