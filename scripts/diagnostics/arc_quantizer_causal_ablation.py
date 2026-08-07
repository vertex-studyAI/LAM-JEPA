from __future__ import annotations

import argparse
import json
import math
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
    "canonical",
    "quantizer_only",
    "no_quantizer",
    "direct_latent",
)


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
        raise RuntimeError(f"unable to construct balanced subset: { {k: len(v) for k, v in buckets.items()} }")
    return [buckets[label][index] for index in range(per_class) for label in range(4)]


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def evaluate(model: LAMARCClassifier, tokens, numeric_x, labels, model_steps: int) -> dict[str, object]:
    model.eval()
    logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    return {
        "accuracy": float(predictions.eq(labels).float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
        "prediction_histogram": {str(k): int(v) for k, v in sorted(Counter(predictions.tolist()).items())},
        "unique_predicted_classes": len(set(predictions.tolist())),
        "mean_max_probability": float(probs.max(dim=-1).values.mean().item()),
        "mean_logit_variance_across_examples": float(logits.float().var(dim=0, unbiased=False).mean().item()),
        "z_feature_std_mean": float(outputs["z"].float().std(dim=0, unbiased=False).mean().item()),
        "z_q_feature_std_mean": float(outputs["z_q"].float().std(dim=0, unbiased=False).mean().item()),
        "latent_summary_feature_std_mean": float(outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item()),
        "quant_loss": float(outputs["quant_loss"].detach().item()),
    }


def config_for(condition: str) -> LAMJEPAConfig:
    base = LAMJEPAConfig()
    if condition == "canonical":
        return base
    if condition == "quantizer_only":
        return replace(base, use_quantizer=True, use_memory=False, use_planner=False)
    if condition == "no_quantizer":
        return replace(base, use_quantizer=False, use_memory=True, use_planner=True)
    if condition == "direct_latent":
        return replace(base, use_quantizer=False, use_memory=False, use_planner=False)
    raise ValueError(condition)


def train_condition(condition: str, tokens, numeric_x, labels, *, seed: int, lr: float, model_steps: int) -> dict[str, object]:
    cfg = config_for(condition)
    set_seed(seed)
    model = LAMARCClassifier(cfg, num_choices=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = [{"step": 0, **evaluate(model, tokens, numeric_x, labels, model_steps)}]

    for step in range(1, max(CHECKPOINTS) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=False)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        total_grad = grad_norm(model.parameters())
        encoder_grad = grad_norm(model.backbone.encoder.parameters())
        projector_grad = grad_norm(model.backbone.projector.parameters())
        quantizer_grad = grad_norm(model.backbone.quantizer.parameters()) if cfg.use_quantizer else 0.0
        choice_head_grad = grad_norm(model.choice_head.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.backbone.update_target()

        if step in CHECKPOINTS:
            history.append(
                {
                    "step": step,
                    "training_loss": float(loss.item()),
                    "gradient_norm_preclip": total_grad,
                    "encoder_gradient_norm_preclip": encoder_grad,
                    "projector_gradient_norm_preclip": projector_grad,
                    "quantizer_gradient_norm_preclip": quantizer_grad,
                    "choice_head_gradient_norm_preclip": choice_head_grad,
                    **evaluate(model, tokens, numeric_x, labels, model_steps),
                }
            )

    return {
        "seed": seed,
        "config": {
            "use_quantizer": cfg.use_quantizer,
            "use_memory": cfg.use_memory,
            "use_planner": cfg.use_planner,
        },
        "history": history,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    final = [float(record["history"][-1]["accuracy"]) for record in records]
    best = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    return {
        "final_accuracy_by_seed": final,
        "best_accuracy_by_seed": best,
        "mean_final_accuracy": float(statistics.fmean(final)),
        "mean_best_accuracy": float(statistics.fmean(best)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best),
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    canonical = bool(summaries["canonical"]["all_seeds_reach_overfit_threshold"])
    quantizer_only = bool(summaries["quantizer_only"]["all_seeds_reach_overfit_threshold"])
    no_quantizer = bool(summaries["no_quantizer"]["all_seeds_reach_overfit_threshold"])
    direct = bool(summaries["direct_latent"]["all_seeds_reach_overfit_threshold"])

    if canonical:
        return "CANONICAL_TRAINABILITY_NOW_PASSES"
    if direct and no_quantizer and not quantizer_only:
        return "QUANTIZER_CAUSAL_COLLAPSE_SUPPORTED"
    if direct and not no_quantizer:
        return "POST_QUANTIZER_MEMORY_OR_PLANNER_FAILURE_REMAINS"
    if not direct:
        return "DIRECT_LATENT_OR_ANSWER_HEAD_TRAINABILITY_FAILURE"
    return "MIXED_OR_SEED_SENSITIVE_QUANTIZER_CAUSAL_RESULT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only causal ablation for the LAM-JEPA ARC quantization boundary.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    args = parser.parse_args()

    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain at least two unique seeds")
    if args.per_class < 1:
        parser.error("--per-class must be positive")

    source = load_arc_split(args.train)
    eligible = select_protocol_eligible_examples(source).eligible
    subset = balanced_subset(eligible, args.per_class)
    label_counts = Counter(example.label for example in subset)
    expected = {label: args.per_class for label in range(4)}
    if dict(label_counts) != expected:
        raise RuntimeError(f"diagnostic subset is not balanced: {dict(label_counts)}")

    base_cfg = LAMJEPAConfig()
    tokens, numeric_x, labels = batchify(subset, vocab_size=base_cfg.vocab_size, device="cpu")
    records: dict[str, list[dict[str, object]]] = {name: [] for name in CONDITIONS}
    for seed in args.seeds:
        for condition in CONDITIONS:
            records[condition].append(
                train_condition(
                    condition,
                    tokens,
                    numeric_x,
                    labels,
                    seed=seed,
                    lr=args.learning_rate,
                    model_steps=args.model_steps,
                )
            )

    summaries = {name: summarize(condition_records) for name, condition_records in records.items()}
    result = {
        "artifact_type": "LAM-JEPA ARC quantizer causal ablation",
        "scope": "training-only diagnostic; no validation or test access",
        "protocol_context": "lam-jepa-arc-challenge-v4",
        "predeclared_question": "Is quantization necessary and/or sufficient for the observed LAM trainability collapse?",
        "subset": {
            "source": "ARC train only",
            "rows": len(subset),
            "per_class": args.per_class,
            "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
            "ids": [example.item_id for example in subset],
        },
        "training": {
            "seeds": args.seeds,
            "steps": max(CHECKPOINTS),
            "checkpoints": list(CHECKPOINTS),
            "learning_rate": args.learning_rate,
            "model_steps": args.model_steps,
            "batch_size": len(subset),
            "overfit_threshold": OVERFIT_THRESHOLD,
        },
        "conditions": records,
        "summaries": summaries,
        "diagnosis": classify(summaries),
        "claim_boundary": {
            "validation_accessed": False,
            "test_accessed": False,
            "performance_claim_authorized": False,
            "mechanism_claim_authorized_beyond_trainability": False,
            "research_complete": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosis": result["diagnosis"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
