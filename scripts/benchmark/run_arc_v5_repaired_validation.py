from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    LAMARCClassifier,
    batchify,
    dataset_digest,
    id_digest,
    iter_minibatches,
    load_arc_split,
)
from lam_jepa.benchmarking.arc_protocol import ARC_PROTOCOL_CHOICE_COUNT, select_protocol_eligible_examples
from lam_jepa.benchmarking.arc_v5_repair import ARC_V5_REPAIR_ID, build_arc_v5_repaired_classifier
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

CONDITIONS = ("legacy_ce", "repaired_v5_ce", "no_quantizer_ce", "repaired_v5_shuffled_labels")
SHUFFLE_SEED_BASE = 20260808
BOOTSTRAP_SAMPLES = 10000


def permute_training_labels(examples: Sequence[ARCExample], *, seed: int) -> list[ARCExample]:
    labels = [example.label for example in examples]
    permuted = list(labels)
    random.Random(seed).shuffle(permuted)
    if len(permuted) > 1 and permuted == labels:
        permuted = permuted[1:] + permuted[:1]
    if sorted(permuted) != sorted(labels):
        raise RuntimeError("label permutation changed the class multiset")
    return [
        ARCExample(item_id=e.item_id, question=e.question, choices=e.choices, label=label)
        for e, label in zip(examples, permuted, strict=True)
    ]


def build_model(condition: str, *, device: str):
    cfg = LAMJEPAConfig()
    if condition == "legacy_ce":
        model = LAMARCClassifier(cfg, num_choices=4)
    elif condition in {"repaired_v5_ce", "repaired_v5_shuffled_labels"}:
        model = build_arc_v5_repaired_classifier(cfg, num_choices=4)
    elif condition == "no_quantizer_ce":
        cfg = replace(cfg, use_quantizer=False)
        model = LAMARCClassifier(cfg, num_choices=4)
    else:
        raise ValueError(condition)
    return cfg, model.to(device)


def train_ce(
    condition: str,
    train: Sequence[ARCExample],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_steps: int,
    device: str,
):
    set_seed(seed)
    cfg, model = build_model(condition, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens, numeric_x, model_steps=model_steps, deterministic=False)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.backbone.update_target()
    return cfg, model.eval()


@torch.no_grad()
def evaluate(model, cfg: LAMJEPAConfig, examples: Sequence[ARCExample], *, batch_size: int, model_steps: int, device: str):
    rows: list[dict[str, object]] = []
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        logits, _ = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
        probs = torch.softmax(logits, dim=-1).cpu()
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.cpu()
        logits_all.append(logits_cpu)
        labels_all.append(labels_cpu)
        for example, probability, label in zip(batch, probs, labels_cpu, strict=True):
            rows.append({
                "id": example.item_id,
                "label": int(label.item()),
                "prediction": int(probability.argmax().item()),
                "probabilities": [float(x) for x in probability.tolist()],
            })
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    predictions = logits.argmax(dim=-1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    histogram = Counter(int(x) for x in predictions.tolist())
    largest_share = max(histogram.values()) / len(rows)
    max_probs = torch.softmax(logits, dim=-1).max(dim=-1).values
    return {
        "accuracy": accuracy,
        "prediction_support": len(histogram),
        "prediction_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        "largest_predicted_class_share": float(largest_share),
        "mean_max_probability": float(max_probs.mean().item()),
        "mean_logit_variance": float(logits.float().var(dim=0, unbiased=False).mean().item()),
        "rows": rows,
    }


def bootstrap_ci(values: Sequence[float], *, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    n = len(values)
    samples = [float(statistics.fmean(values[rng.randrange(n)] for _ in range(n))) for _ in range(BOOTSTRAP_SAMPLES)]
    samples.sort()
    return samples[int(0.025 * (BOOTSTRAP_SAMPLES - 1))], samples[int(0.975 * (BOOTSTRAP_SAMPLES - 1))]


def summarize(values: Sequence[float], *, seed: int) -> dict[str, object]:
    low, high = bootstrap_ci(values, seed=seed)
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "by_seed": [float(x) for x in values],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen ARC-v5 repaired validation protocol without ARC test access.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_VALIDATION_EXECUTION":
        raise SystemExit("validation protocol is not frozen")
    if protocol.get("repair_id") != ARC_V5_REPAIR_ID:
        raise SystemExit("frozen repair id does not match implementation")
    training = protocol["training"]
    seeds = [int(x) for x in training["seeds"]]
    if seeds != [1, 2, 3, 4, 5]:
        raise SystemExit("unexpected frozen seed set")

    train_source = load_arc_split(args.train)
    validation_source = load_arc_split(args.validation)
    train_partition = select_protocol_eligible_examples(train_source)
    validation_partition = select_protocol_eligible_examples(validation_source)
    train = list(train_partition.eligible)
    validation = list(validation_partition.eligible)
    if len(train) != int(protocol["dataset"]["train_eligible_rows"]):
        raise SystemExit("eligible train-row count differs from frozen protocol")
    if len(validation) != int(protocol["dataset"]["validation_eligible_rows"]):
        raise SystemExit("eligible validation-row count differs from frozen protocol")
    if any(len(e.choices) != ARC_PROTOCOL_CHOICE_COUNT for e in train + validation):
        raise SystemExit("non-four-choice row entered frozen eligible set")
    if {e.item_id for e in train} & {e.item_id for e in validation}:
        raise SystemExit("train/validation id leakage")

    records: dict[str, list[dict[str, object]]] = {condition: [] for condition in CONDITIONS}
    for seed in seeds:
        shuffled = permute_training_labels(train, seed=SHUFFLE_SEED_BASE + seed)
        for condition in CONDITIONS:
            condition_train = shuffled if condition == "repaired_v5_shuffled_labels" else train
            cfg, model = train_ce(
                condition,
                condition_train,
                seed=seed,
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                model_steps=int(training["model_steps"]),
                device=args.device,
            )
            result = evaluate(
                model,
                cfg,
                validation,
                batch_size=int(training["batch_size"]),
                model_steps=int(training["model_steps"]),
                device=args.device,
            )
            records[condition].append({"seed": seed, **result})
            del model

    accuracy = {condition: [float(r["accuracy"]) for r in records[condition]] for condition in CONDITIONS}
    repaired_minus_legacy = [r - l for r, l in zip(accuracy["repaired_v5_ce"], accuracy["legacy_ce"], strict=True)]
    repaired_minus_noq = [r - n for r, n in zip(accuracy["repaired_v5_ce"], accuracy["no_quantizer_ce"], strict=True)]
    summaries = {
        condition: summarize(values, seed=SHUFFLE_SEED_BASE + index)
        for index, (condition, values) in enumerate(accuracy.items())
    }
    summaries["repaired_minus_legacy"] = summarize(repaired_minus_legacy, seed=SHUFFLE_SEED_BASE + 20)
    summaries["repaired_minus_no_quantizer"] = summarize(repaired_minus_noq, seed=SHUFFLE_SEED_BASE + 21)

    negative_control_valid = float(summaries["repaired_v5_shuffled_labels"]["bootstrap_ci95_high"]) < 0.35
    collapse_rejected = all(
        int(row["prediction_support"]) >= 2 and float(row["largest_predicted_class_share"]) <= 0.95
        for row in records["repaired_v5_ce"]
    )
    generalization_supported = bool(
        negative_control_valid
        and collapse_rejected
        and float(summaries["repaired_v5_ce"]["bootstrap_ci95_low"]) > 0.25
        and float(summaries["repaired_minus_legacy"]["bootstrap_ci95_low"]) > 0.0
    )
    quantization_benefit_supported = bool(float(summaries["repaired_minus_no_quantizer"]["bootstrap_ci95_low"]) > 0.0)
    if not negative_control_valid:
        verdict = "INVALID_NEGATIVE_CONTROL"
    elif generalization_supported:
        verdict = "VALIDATION_GENERALIZATION_SUPPORTED_WITH_LIMITATIONS"
    else:
        verdict = "VALID_NEGATIVE_OR_INCONCLUSIVE_VALIDATION"

    payload = {
        "artifact_type": "LAM-JEPA ARC v5 repaired validation result package",
        "protocol": protocol,
        "dataset_evidence": {
            "train_source_rows": len(train_source),
            "validation_source_rows": len(validation_source),
            "train_eligible_rows": len(train),
            "validation_eligible_rows": len(validation),
            "train_dataset_digest": dataset_digest(train),
            "validation_dataset_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
        },
        "records": records,
        "summaries": summaries,
        "decision_rules": {
            "negative_control_valid": negative_control_valid,
            "collapse_rejected": collapse_rejected,
            "generalization_supported_with_limitations": generalization_supported,
            "quantization_benefit_supported": quantization_benefit_supported,
        },
        "verdict": verdict,
        "claim_boundary": {
            "validation_accessed": True,
            "test_accessed": False,
            "confirmatory_test_claim_authorized": False,
            "external_generalization_claim_authorized": False,
            "research_complete": False,
            "independent_result_reproduction_required": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "summaries": summaries, "decision_rules": payload["decision_rules"]}, indent=2))
    if not negative_control_valid:
        raise SystemExit("validation package invalid because shuffled-label negative control failed")


if __name__ == "__main__":
    main()
