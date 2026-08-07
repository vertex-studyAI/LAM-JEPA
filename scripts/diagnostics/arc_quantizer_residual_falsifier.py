from __future__ import annotations

import argparse
import json
import statistics
import types
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
    "stable_hard_1_00",
    "stable_mix_0_75",
    "stable_mix_0_50",
    "stable_mix_0_25",
    "no_quantizer",
)
HARD_FRACTIONS = {
    "stable_hard_1_00": 1.0,
    "stable_mix_0_75": 0.75,
    "stable_mix_0_50": 0.50,
    "stable_mix_0_25": 0.25,
}


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
        raise RuntimeError("unable to construct balanced four-class subset")
    return [buckets[label][index] for index in range(per_class) for label in range(4)]


def stabilize_ema(model: LAMARCClassifier) -> None:
    q = model.backbone.quantizer
    with torch.no_grad():
        q.ema_count.fill_(1.0)
        q.ema_weight.copy_(q.codebook.detach())


def install_residual_mix(model: LAMARCClassifier, hard_fraction: float) -> None:
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0, 1]")
    q = model.backbone.quantizer
    original_forward = q.forward

    def residual_forward(self, z: torch.Tensor):
        hard_z_q, quant_loss, indices = original_forward(z)
        mixed_z_q = z + hard_fraction * (hard_z_q - z)
        return mixed_z_q, quant_loss, indices

    q.forward = types.MethodType(residual_forward, q)


def quantizer_state(model: LAMARCClassifier, indices: list[int] | None) -> dict[str, object]:
    if not model.backbone.cfg.use_quantizer:
        return {"enabled": False, "code_support": None, "code_histogram": {}}
    q = model.backbone.quantizer
    code_norms = q.codebook.detach().float().norm(dim=1)
    result: dict[str, object] = {
        "enabled": True,
        "code_support": len(set(indices or [])),
        "code_histogram": {str(k): int(v) for k, v in sorted(Counter(indices or []).items())},
        "codebook_max_norm": float(code_norms.max().item()),
        "codebook_mean_norm": float(code_norms.mean().item()),
        "ema_count_min": float(q.ema_count.min().item()),
        "ema_count_max": float(q.ema_count.max().item()),
        "ema_count_mean": float(q.ema_count.mean().item()),
        "finite": bool(
            torch.isfinite(q.codebook).all()
            and torch.isfinite(q.ema_count).all()
            and torch.isfinite(q.ema_weight).all()
        ),
    }
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
        "latent_summary_feature_std": float(
            outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item()
        ),
        "quantizer_state": quantizer_state(model, indices),
    }


def train_condition(condition: str, seed: int, tokens, numeric_x, labels) -> dict[str, object]:
    set_seed(seed)
    if condition == "no_quantizer":
        cfg = replace(LAMJEPAConfig(), use_quantizer=False)
        hard_fraction = None
    else:
        cfg = LAMJEPAConfig()
        hard_fraction = HARD_FRACTIONS[condition]

    model = LAMARCClassifier(cfg, num_choices=4)
    if cfg.use_quantizer:
        stabilize_ema(model)
        install_residual_mix(model, hard_fraction)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = [{"step": 0, **evaluate(model, tokens, numeric_x, labels)}]

    for step in range(1, 301):
        model.train()
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
        "hard_fraction": hard_fraction,
        "continuous_residual_fraction": None if hard_fraction is None else 1.0 - hard_fraction,
        "history": history,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    best = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    final = [float(record["history"][-1]["accuracy"]) for record in records]
    return {
        "best_accuracy_by_seed": best,
        "final_accuracy_by_seed": final,
        "mean_best_accuracy": float(statistics.fmean(best)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best),
    }


def classify(summaries: dict[str, dict[str, object]]) -> tuple[str, str | None]:
    passing_mixes = [
        condition
        for condition in HARD_FRACTIONS
        if bool(summaries[condition]["all_seeds_reach_overfit_threshold"])
    ]
    if passing_mixes:
        selected = max(passing_mixes, key=lambda name: HARD_FRACTIONS[name])
        if HARD_FRACTIONS[selected] == 1.0:
            return "STABLE_EMA_HARD_QUANTIZATION_NOW_PASSES", selected
        return "CONTROLLED_CONTINUOUS_RESIDUAL_RESTORES_TRAINABILITY", selected
    if bool(summaries["no_quantizer"]["all_seeds_reach_overfit_threshold"]):
        return "DISCRETE_FORWARD_COMPONENT_REMAINS_BLOCKING_ACROSS_RESIDUAL_LADDER", None
    return "TRAINABILITY_REFERENCE_DRIFT_OR_MIXED_FAILURE", None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only residual-preservation falsifier for the ARC quantization boundary.")
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
    diagnosis, selected = classify(summaries)
    payload = {
        "artifact_type": "LAM-JEPA ARC quantizer residual falsifier",
        "scope": "training-only diagnostic; no validation or test access",
        "question": "Does preserving a fixed continuous residual through the quantization boundary restore reliable trainability while retaining a hard-code contribution?",
        "subset": {"rows": 32, "per_class": 8, "ids": [example.item_id for example in subset]},
        "training": {
            "seeds": [1, 2],
            "steps": 300,
            "learning_rate": 3e-4,
            "overfit_threshold": OVERFIT_THRESHOLD,
        },
        "conditions": {
            "stable_hard_1_00": "stable EMA state; 100% hard selected-code forward contribution and 0% continuous residual",
            "stable_mix_0_75": "stable EMA state; 75% hard-code displacement and 25% continuous residual",
            "stable_mix_0_50": "stable EMA state; 50% hard-code displacement and 50% continuous residual",
            "stable_mix_0_25": "stable EMA state; 25% hard-code displacement and 75% continuous residual",
            "no_quantizer": "quantization disabled while memory and planner remain enabled; positive trainability reference",
        },
        "records": records,
        "summaries": summaries,
        "diagnosis": diagnosis,
        "minimal_passing_residual_condition": selected,
        "selection_rule": "among quantized residual conditions passing both seeds, select the highest hard_fraction / smallest continuous residual; never select using validation",
        "claim_boundary": {
            "validation_accessed": False,
            "test_accessed": False,
            "performance_claim_authorized": False,
            "research_complete": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnosis": diagnosis,
        "minimal_passing_residual_condition": selected,
        "summaries": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
