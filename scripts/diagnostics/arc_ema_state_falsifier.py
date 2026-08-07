from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, batchify, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 200, 300)
OVERFIT_THRESHOLD = 0.95
CONDITIONS = ("canonical", "count_only", "weight_only", "synchronized")


def balanced_subset(examples, per_class: int) -> list:
    buckets = {label: [] for label in range(4)}
    for example in examples:
        if len(example.choices) != 4:
            continue
        if len(buckets[example.label]) < per_class:
            buckets[example.label].append(example)
        if all(len(bucket) == per_class for bucket in buckets.values()):
            break
    if not all(len(bucket) == per_class for bucket in buckets.values()):
        raise RuntimeError("unable to construct balanced four-class subset")
    return [buckets[label][index] for index in range(per_class) for label in range(4)]


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
    }


def quantizer_state(model: LAMARCClassifier) -> dict[str, object]:
    q = model.backbone.quantizer
    code_norms = q.codebook.detach().float().norm(dim=1)
    ema_weight_norms = q.ema_weight.detach().float().norm(dim=1)
    return {
        "codebook_norm": tensor_stats(code_norms),
        "ema_count": tensor_stats(q.ema_count),
        "ema_weight_norm": tensor_stats(ema_weight_norms),
        "finite": bool(
            torch.isfinite(q.codebook).all()
            and torch.isfinite(q.ema_count).all()
            and torch.isfinite(q.ema_weight).all()
        ),
    }


def apply_state_condition(model: LAMARCClassifier, condition: str) -> None:
    q = model.backbone.quantizer
    with torch.no_grad():
        if condition in {"count_only", "synchronized"}:
            q.ema_count.fill_(1.0)
        if condition in {"weight_only", "synchronized"}:
            q.ema_weight.copy_(q.codebook.detach())


def evaluate(model: LAMARCClassifier, tokens, numeric_x, labels) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        logits, outputs = model(tokens, numeric_x, model_steps=1, deterministic=True)
        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        indices = outputs["indices"].detach().cpu().tolist()
        return {
            "accuracy": float(predictions.eq(labels).float().mean().item()),
            "cross_entropy": float(F.cross_entropy(logits, labels).item()),
            "prediction_support": len(set(predictions.detach().cpu().tolist())),
            "code_support": len(set(indices)),
            "code_histogram": {str(k): int(v) for k, v in sorted(Counter(indices).items())},
            "z_feature_std": float(outputs["z"].float().std(dim=0, unbiased=False).mean().item()),
            "z_q_feature_std": float(outputs["z_q"].float().std(dim=0, unbiased=False).mean().item()),
            "latent_summary_feature_std": float(
                outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item()
            ),
            "quantizer_state": quantizer_state(model),
        }


def train_condition(condition: str, seed: int, tokens, numeric_x, labels) -> dict[str, object]:
    set_seed(seed)
    model = LAMARCClassifier(LAMJEPAConfig(), num_choices=4)
    apply_state_condition(model, condition)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    record: dict[str, object] = {
        "condition": condition,
        "seed": seed,
        "state_after_condition_init": quantizer_state(model),
        "history": [{"step": 0, **evaluate(model, tokens, numeric_x, labels)}],
    }

    for step in range(1, max(CHECKPOINTS) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, outputs = model(tokens, numeric_x, model_steps=1, deterministic=False)
        loss = F.cross_entropy(logits, labels)

        if step == 1:
            record["first_training_forward"] = {
                "loss": float(loss.detach().item()),
                "code_support": len(set(outputs["indices"].detach().cpu().tolist())),
                "z_feature_std": float(outputs["z"].detach().float().std(dim=0, unbiased=False).mean().item()),
                "z_q_feature_std": float(outputs["z_q"].detach().float().std(dim=0, unbiased=False).mean().item()),
                "latent_summary_feature_std": float(
                    outputs["latent_summary"].detach().float().std(dim=0, unbiased=False).mean().item()
                ),
                "quantizer_state_after_forward_before_optimizer": quantizer_state(model),
            }

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.backbone.update_target()

        if step in CHECKPOINTS:
            record["history"].append({"step": step, **evaluate(model, tokens, numeric_x, labels)})

    return record


def summary(records: list[dict[str, object]]) -> dict[str, object]:
    best = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    final = [float(record["history"][-1]["accuracy"]) for record in records]
    first_codebook_max = [
        float(record["first_training_forward"]["quantizer_state_after_forward_before_optimizer"]["codebook_norm"]["max"])
        for record in records
    ]
    first_code_support = [int(record["first_training_forward"]["code_support"]) for record in records]
    return {
        "best_accuracy_by_seed": best,
        "final_accuracy_by_seed": final,
        "mean_best_accuracy": float(statistics.fmean(best)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best),
        "first_forward_codebook_max_norm_by_seed": first_codebook_max,
        "first_forward_code_support_by_seed": first_code_support,
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    sync_pass = bool(summaries["synchronized"]["all_seeds_reach_overfit_threshold"])
    count_pass = bool(summaries["count_only"]["all_seeds_reach_overfit_threshold"])
    weight_pass = bool(summaries["weight_only"]["all_seeds_reach_overfit_threshold"])
    canonical_pass = bool(summaries["canonical"]["all_seeds_reach_overfit_threshold"])
    if canonical_pass:
        return "CANONICAL_FAILURE_NOT_REPRODUCED"
    if sync_pass and count_pass and not weight_pass:
        return "EMA_COUNT_PSEUDOCOUNT_SUFFICIENT_WITH_OR_WITHOUT_WEIGHT_SYNC"
    if sync_pass and not count_pass and not weight_pass:
        return "SYNCHRONIZED_EMA_INITIALIZATION_REQUIRED"
    if sync_pass:
        return "EMA_STATE_INITIALIZATION_REPAIR_SUPPORTED_WITH_MIXED_COMPONENT_EFFECTS"
    return "EMA_STATE_INITIALIZATION_NOT_SUFFICIENT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only falsifier for EMA quantizer initialization state.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    eligible = select_protocol_eligible_examples(load_arc_split(args.train)).eligible
    subset = balanced_subset(eligible, 8)
    cfg = LAMJEPAConfig()
    tokens, numeric_x, labels = batchify(subset, vocab_size=cfg.vocab_size, device="cpu")

    records: dict[str, list[dict[str, object]]] = {condition: [] for condition in CONDITIONS}
    for seed in (1, 2):
        for condition in CONDITIONS:
            records[condition].append(train_condition(condition, seed, tokens, numeric_x, labels))

    summaries = {condition: summary(records[condition]) for condition in CONDITIONS}
    payload = {
        "artifact_type": "LAM-JEPA ARC EMA state falsifier",
        "scope": "training-only diagnostic; no validation or test access",
        "question": "Does coherent EMA quantizer initialization prevent the first-update representation collapse?",
        "subset": {
            "rows": 32,
            "per_class": 8,
            "ids": [example.item_id for example in subset],
        },
        "training": {
            "seeds": [1, 2],
            "steps": 300,
            "learning_rate": 3e-4,
            "overfit_threshold": OVERFIT_THRESHOLD,
        },
        "conditions": {
            "canonical": "existing zero ema_count plus independent random ema_weight",
            "count_only": "ema_count initialized to one; existing independent random ema_weight retained",
            "weight_only": "ema_weight synchronized to initial codebook; zero ema_count retained",
            "synchronized": "ema_count initialized to one and ema_weight synchronized to initial codebook",
        },
        "records": records,
        "summaries": summaries,
        "diagnosis": classify(summaries),
        "claim_boundary": {
            "validation_accessed": False,
            "test_accessed": False,
            "performance_claim_authorized": False,
            "research_complete": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosis": payload["diagnosis"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
