from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, batchify, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 101, 150, 200, 300)
OVERFIT_THRESHOLD = 0.95
CONDITIONS = ("canonical", "stable_ema", "frozen_codebook", "delayed_100")


def balanced_subset(examples, per_class: int = 8) -> list:
    buckets = {label: [] for label in range(4)}
    for example in examples:
        if len(example.choices) != 4:
            continue
        if len(buckets[example.label]) < per_class:
            buckets[example.label].append(example)
        if all(len(bucket) == per_class for bucket in buckets.values()):
            break
    if not all(len(bucket) == per_class for bucket in buckets.values()):
        raise RuntimeError("unable to construct balanced subset")
    return [buckets[label][index] for index in range(per_class) for label in range(4)]


def initialize_stable_ema(model: LAMARCClassifier) -> None:
    q = model.backbone.quantizer
    with torch.no_grad():
        q.ema_count.fill_(1.0)
        q.ema_weight.copy_(q.codebook.detach())


def state(model: LAMARCClassifier) -> dict[str, object]:
    q = model.backbone.quantizer
    norms = q.codebook.detach().float().norm(dim=1)
    return {
        "codebook_max_norm": float(norms.max().item()),
        "codebook_mean_norm": float(norms.mean().item()),
        "ema_count_min": float(q.ema_count.min().item()),
        "ema_count_max": float(q.ema_count.max().item()),
        "ema_count_mean": float(q.ema_count.mean().item()),
        "finite": bool(torch.isfinite(q.codebook).all() and torch.isfinite(q.ema_count).all() and torch.isfinite(q.ema_weight).all()),
    }


def evaluate(model: LAMARCClassifier, tokens, numeric_x, labels) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        logits, outputs = model(tokens, numeric_x, model_steps=1, deterministic=True)
        predictions = logits.argmax(dim=-1)
        indices = outputs["indices"].detach().cpu().tolist()
        return {
            "accuracy": float(predictions.eq(labels).float().mean().item()),
            "cross_entropy": float(F.cross_entropy(logits, labels).item()),
            "prediction_support": len(set(predictions.detach().cpu().tolist())),
            "code_support": len(set(indices)),
            "code_histogram": {str(k): int(v) for k, v in sorted(Counter(indices).items())},
            "z_feature_std": float(outputs["z"].float().std(dim=0, unbiased=False).mean().item()),
            "z_q_feature_std": float(outputs["z_q"].float().std(dim=0, unbiased=False).mean().item()),
            "latent_summary_feature_std": float(outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item()),
            "quantizer_state": state(model),
        }


def should_update_quantizer(condition: str, step: int) -> bool:
    if condition in {"canonical", "stable_ema"}:
        return True
    if condition == "frozen_codebook":
        return False
    if condition == "delayed_100":
        return step > 100
    raise ValueError(condition)


def train_condition(condition: str, seed: int, tokens, numeric_x, labels) -> dict[str, object]:
    set_seed(seed)
    model = LAMARCClassifier(LAMJEPAConfig(), num_choices=4)
    if condition in {"stable_ema", "delayed_100"}:
        initialize_stable_ema(model)
    initial_codebook = model.backbone.quantizer.codebook.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = [{"step": 0, "quantizer_update_enabled": False, **evaluate(model, tokens, numeric_x, labels)}]

    for step in range(1, 301):
        model.train()
        update_enabled = should_update_quantizer(condition, step)
        if not update_enabled:
            model.backbone.quantizer.eval()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens, numeric_x, model_steps=1, deterministic=False)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.backbone.update_target()

        if step in CHECKPOINTS:
            current = model.backbone.quantizer.codebook.detach()
            history.append(
                {
                    "step": step,
                    "quantizer_update_enabled": update_enabled,
                    "codebook_l2_change_from_init": float((current - initial_codebook).float().norm().item()),
                    **evaluate(model, tokens, numeric_x, labels),
                }
            )

    return {"condition": condition, "seed": seed, "history": history}


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    best = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    final = [float(record["history"][-1]["accuracy"]) for record in records]
    final_support = [int(record["history"][-1]["code_support"]) for record in records]
    return {
        "best_accuracy_by_seed": best,
        "final_accuracy_by_seed": final,
        "mean_best_accuracy": float(statistics.fmean(best)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best),
        "final_code_support_by_seed": final_support,
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    canonical = bool(summaries["canonical"]["all_seeds_reach_overfit_threshold"])
    static = bool(summaries["frozen_codebook"]["all_seeds_reach_overfit_threshold"])
    stable = bool(summaries["stable_ema"]["all_seeds_reach_overfit_threshold"])
    delayed = bool(summaries["delayed_100"]["all_seeds_reach_overfit_threshold"])
    if canonical:
        return "CANONICAL_FAILURE_NOT_REPRODUCED"
    if static and not stable:
        return "EMA_CODEBOOK_MUTATION_CAUSAL_BLOCKER_SUPPORTED"
    if delayed and not stable:
        return "EARLY_EMA_UPDATE_TIMING_CAUSAL_BLOCKER_SUPPORTED"
    if static and stable:
        return "QUANTIZED_TRAINABILITY_RESTORED_UNDER_MULTIPLE_POLICIES"
    if not static:
        return "STATIC_QUANTIZATION_BOTTLENECK_REMAINS"
    return "MIXED_UPDATE_POLICY_RESULT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only quantizer update-policy falsifier.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    eligible = select_protocol_eligible_examples(load_arc_split(args.train)).eligible
    subset = balanced_subset(eligible)
    cfg = LAMJEPAConfig()
    tokens, numeric_x, labels = batchify(subset, vocab_size=cfg.vocab_size, device="cpu")

    records = {condition: [] for condition in CONDITIONS}
    for seed in (1, 2):
        for condition in CONDITIONS:
            records[condition].append(train_condition(condition, seed, tokens, numeric_x, labels))

    summaries = {condition: summarize(records[condition]) for condition in CONDITIONS}
    payload = {
        "artifact_type": "LAM-JEPA ARC quantizer update-policy falsifier",
        "scope": "training-only diagnostic; no validation or test access",
        "question": "Does EMA codebook mutation, rather than static discrete quantization, cause the frozen ARC trainability failure?",
        "subset": {"rows": 32, "per_class": 8, "ids": [example.item_id for example in subset]},
        "training": {"seeds": [1, 2], "steps": 300, "learning_rate": 3e-4, "overfit_threshold": OVERFIT_THRESHOLD},
        "conditions": {
            "canonical": "existing quantizer initialization and EMA update every step",
            "stable_ema": "unit pseudocounts plus synchronized EMA weights; EMA update every step",
            "frozen_codebook": "canonical initial codebook; quantizer kept in eval mode during optimization so EMA state/codebook never mutate",
            "delayed_100": "stable EMA state; quantizer frozen for steps 1-100 then EMA updates enabled for steps 101-300",
        },
        "records": records,
        "summaries": summaries,
        "diagnosis": classify(summaries),
        "claim_boundary": {"validation_accessed": False, "test_accessed": False, "performance_claim_authorized": False, "research_complete": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosis": payload["diagnosis"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
