from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, batchify, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 200, 300)
OVERFIT_THRESHOLD = 0.95
CONDITIONS = (
    "random32_frozen",
    "random128_frozen",
    "data32_frozen",
    "data32_stable_ema",
    "no_quantizer",
)


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


def config_for(condition: str) -> LAMJEPAConfig:
    base = LAMJEPAConfig()
    if condition == "random128_frozen":
        return replace(base, num_codes=128)
    if condition == "no_quantizer":
        return replace(base, use_quantizer=False)
    return base


def initialize_from_train_latents(model: LAMARCClassifier, tokens, numeric_x) -> None:
    q = model.backbone.quantizer
    require_codes = q.num_codes
    if require_codes != tokens.size(0):
        raise RuntimeError(f"data initialization requires one code per row; got {require_codes} codes for {tokens.size(0)} rows")
    model.eval()
    with torch.no_grad():
        z = model.backbone.encoder(tokens, numeric_x=numeric_x)
        z = model.backbone.projector(z)
        q.codebook.copy_(z)
        q.ema_count.fill_(1.0)
        q.ema_weight.copy_(z)


def stabilize_ema(model: LAMARCClassifier) -> None:
    q = model.backbone.quantizer
    with torch.no_grad():
        q.ema_count.fill_(1.0)
        q.ema_weight.copy_(q.codebook.detach())


def quantizer_state(model: LAMARCClassifier, indices: list[int] | None = None) -> dict[str, object]:
    if not model.backbone.cfg.use_quantizer:
        return {"enabled": False, "code_support": None, "code_histogram": {}}
    q = model.backbone.quantizer
    norms = q.codebook.detach().float().norm(dim=1)
    result: dict[str, object] = {
        "enabled": True,
        "num_codes": q.num_codes,
        "codebook_max_norm": float(norms.max().item()),
        "codebook_mean_norm": float(norms.mean().item()),
        "ema_count_min": float(q.ema_count.min().item()),
        "ema_count_max": float(q.ema_count.max().item()),
        "finite": bool(torch.isfinite(q.codebook).all() and torch.isfinite(q.ema_count).all() and torch.isfinite(q.ema_weight).all()),
    }
    if indices is not None:
        result["code_support"] = len(set(indices))
        result["code_histogram"] = {str(k): int(v) for k, v in sorted(Counter(indices).items())}
    return result


@torch.no_grad()
def evaluate(model: LAMARCClassifier, tokens, numeric_x, labels) -> dict[str, object]:
    model.eval()
    logits, outputs = model(tokens, numeric_x, model_steps=1, deterministic=True)
    predictions = logits.argmax(dim=-1)
    indices = outputs["indices"].detach().cpu().tolist() if model.backbone.cfg.use_quantizer else None
    return {
        "accuracy": float(predictions.eq(labels).float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
        "prediction_support": len(set(predictions.detach().cpu().tolist())),
        "z_feature_std": float(outputs["z"].float().std(dim=0, unbiased=False).mean().item()),
        "z_q_feature_std": float(outputs["z_q"].float().std(dim=0, unbiased=False).mean().item()),
        "latent_summary_feature_std": float(outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item()),
        "quantizer_state": quantizer_state(model, indices),
    }


def train_condition(condition: str, seed: int, tokens, numeric_x, labels) -> dict[str, object]:
    set_seed(seed)
    cfg = config_for(condition)
    model = LAMARCClassifier(cfg, num_choices=4)

    if condition.startswith("data32"):
        initialize_from_train_latents(model, tokens, numeric_x)
    elif condition == "random32_frozen" or condition == "random128_frozen":
        pass
    elif condition == "no_quantizer":
        pass
    else:
        raise ValueError(condition)

    if condition == "data32_stable_ema":
        stabilize_ema(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = [{"step": 0, **evaluate(model, tokens, numeric_x, labels)}]

    frozen_quantizer = condition in {"random32_frozen", "random128_frozen", "data32_frozen"}
    for step in range(1, 301):
        model.train()
        if frozen_quantizer and cfg.use_quantizer:
            model.backbone.quantizer.eval()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens, numeric_x, model_steps=1, deterministic=False)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.backbone.update_target()

        if step in CHECKPOINTS:
            history.append({"step": step, **evaluate(model, tokens, numeric_x, labels)})

    return {
        "condition": condition,
        "seed": seed,
        "config": {"num_codes": cfg.num_codes, "use_quantizer": cfg.use_quantizer},
        "history": history,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    best = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    final = [float(record["history"][-1]["accuracy"]) for record in records]
    final_code_support = [record["history"][-1]["quantizer_state"].get("code_support") for record in records]
    return {
        "best_accuracy_by_seed": best,
        "final_accuracy_by_seed": final,
        "mean_best_accuracy": float(statistics.fmean(best)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best),
        "final_code_support_by_seed": final_code_support,
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    base = bool(summaries["random32_frozen"]["all_seeds_reach_overfit_threshold"])
    larger = bool(summaries["random128_frozen"]["all_seeds_reach_overfit_threshold"])
    data_frozen = bool(summaries["data32_frozen"]["all_seeds_reach_overfit_threshold"])
    data_ema = bool(summaries["data32_stable_ema"]["all_seeds_reach_overfit_threshold"])
    continuous = bool(summaries["no_quantizer"]["all_seeds_reach_overfit_threshold"])
    if base:
        return "STATIC_RANDOM32_FAILURE_NOT_REPRODUCED"
    if data_frozen and not larger:
        return "CODEBOOK_INITIALIZATION_IS_PRIMARY_STATIC_ASSIGNMENT_BLOCKER"
    if larger and not data_frozen:
        return "CODEBOOK_CAPACITY_IS_PRIMARY_STATIC_ASSIGNMENT_BLOCKER"
    if data_frozen and larger:
        return "CAPACITY_AND_INITIALIZATION_BOTH_RESTORE_STATIC_TRAINABILITY"
    if data_ema and not data_frozen:
        return "EMA_ADAPTATION_WITH_DATA_INITIALIZATION_RESTORES_TRAINABILITY"
    if continuous and not data_frozen and not larger:
        return "DISCRETE_ASSIGNMENT_BOTTLENECK_REMAINS_BEYOND_SIZE_AND_INITIALIZATION"
    return "MIXED_CODEBOOK_ASSIGNMENT_RESULT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only codebook capacity/initialization falsifier for ARC.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    eligible = select_protocol_eligible_examples(load_arc_split(args.train)).eligible
    subset = balanced_subset(eligible)
    base_cfg = LAMJEPAConfig()
    tokens, numeric_x, labels = batchify(subset, vocab_size=base_cfg.vocab_size, device="cpu")

    records = {condition: [] for condition in CONDITIONS}
    for seed in (1, 2):
        for condition in CONDITIONS:
            records[condition].append(train_condition(condition, seed, tokens, numeric_x, labels))

    summaries = {condition: summarize(records[condition]) for condition in CONDITIONS}
    payload = {
        "artifact_type": "LAM-JEPA ARC codebook assignment falsifier",
        "scope": "training-only diagnostic; no validation or test access",
        "question": "Is the remaining static quantization failure primarily codebook capacity, codebook initialization, or a deeper discrete-assignment bottleneck?",
        "subset": {"rows": 32, "per_class": 8, "ids": [example.item_id for example in subset]},
        "training": {"seeds": [1, 2], "steps": 300, "learning_rate": 3e-4, "overfit_threshold": OVERFIT_THRESHOLD},
        "conditions": {
            "random32_frozen": "current 32-code random codebook, static during optimization",
            "random128_frozen": "128-code random codebook, static during optimization; capacity only changes",
            "data32_frozen": "32-code static codebook initialized one-to-one from the 32 projected train latents at step 0",
            "data32_stable_ema": "same train-latent initialization with unit EMA counts and synchronized EMA weights; EMA remains active",
            "no_quantizer": "quantization disabled while memory and planner remain enabled; positive trainability reference",
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
