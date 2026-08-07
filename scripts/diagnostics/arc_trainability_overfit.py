from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, _lam_arc_loss, batchify, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed
from run_arc_matched_baseline import (
    MatchedSupervisedClassifier,
    choose_matched_architecture,
    probe_lam_active_parameters,
)

CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 200, 300)
OVERFIT_THRESHOLD = 0.95


def balanced_subset(examples, per_class: int = 8):
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
    selected = []
    for index in range(per_class):
        for label in range(4):
            selected.append(buckets[label][index])
    return selected


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def common_eval(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    probs = torch.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    return {
        "accuracy": float(predictions.eq(labels).float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
        "prediction_histogram": {str(k): int(v) for k, v in sorted(Counter(predictions.cpu().tolist()).items())},
        "unique_predicted_classes": len(set(predictions.cpu().tolist())),
        "mean_max_probability": float(probs.max(dim=-1).values.mean().item()),
        "mean_logit_variance_across_examples": float(logits.float().var(dim=0, unbiased=False).mean().item()),
    }


@torch.no_grad()
def eval_lam(model, tokens, numeric_x, labels, model_steps: int) -> dict[str, object]:
    model.eval()
    logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
    result = common_eval(logits, labels)
    result["latent_summary_feature_std_mean"] = float(outputs["latent_summary"].float().std(dim=0, unbiased=False).mean().item())
    result["z_feature_std_mean"] = float(outputs["z"].float().std(dim=0, unbiased=False).mean().item())
    result["z_q_feature_std_mean"] = float(outputs["z_q"].float().std(dim=0, unbiased=False).mean().item())
    return result


@torch.no_grad()
def eval_matched(model, tokens, numeric_x, labels) -> dict[str, object]:
    model.eval()
    return common_eval(model(tokens, numeric_x), labels)


def train_matched(cfg, tokens, numeric_x, labels, *, seed: int, hidden_dim: int, depth: int, lr: float):
    set_seed(seed)
    model = MatchedSupervisedClassifier(cfg, hidden_dim=hidden_dim, depth=depth, num_choices=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = [{"step": 0, **eval_matched(model, tokens, numeric_x, labels)}]
    last_grad = None
    for step in range(1, max(CHECKPOINTS) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, numeric_x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        last_grad = grad_norm(model.parameters())
        optimizer.step()
        if step in CHECKPOINTS:
            history.append({"step": step, "training_loss": float(loss.item()), "gradient_norm": last_grad, **eval_matched(model, tokens, numeric_x, labels)})
    return history


def train_lam(cfg, tokens, numeric_x, labels, *, seed: int, lr: float, model_steps: int, full_objective: bool):
    set_seed(seed)
    model = LAMARCClassifier(cfg, num_choices=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = [{"step": 0, **eval_lam(model, tokens, numeric_x, labels, model_steps)}]
    for step in range(1, max(CHECKPOINTS) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=False)
        supervised = F.cross_entropy(logits, labels)
        total = _lam_arc_loss(logits, outputs, labels) if full_objective else supervised
        auxiliary = total - supervised
        total.backward()
        total_grad = grad_norm(model.parameters())
        choice_head_grad = grad_norm(model.choice_head.parameters())
        encoder_grad = grad_norm(model.backbone.encoder.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.backbone.update_target()
        if step in CHECKPOINTS:
            history.append(
                {
                    "step": step,
                    "training_total_loss": float(total.item()),
                    "training_supervised_loss": float(supervised.item()),
                    "training_auxiliary_loss": float(auxiliary.item()),
                    "gradient_norm_preclip": total_grad,
                    "choice_head_gradient_norm_preclip": choice_head_grad,
                    "encoder_gradient_norm_preclip": encoder_grad,
                    **eval_lam(model, tokens, numeric_x, labels, model_steps),
                }
            )
    return history


def summarize_condition(records: list[dict[str, object]]) -> dict[str, object]:
    final_accuracies = [float(record["history"][-1]["accuracy"]) for record in records]
    best_accuracies = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    return {
        "final_accuracy_by_seed": final_accuracies,
        "best_accuracy_by_seed": best_accuracies,
        "mean_final_accuracy": float(statistics.fmean(final_accuracies)),
        "mean_best_accuracy": float(statistics.fmean(best_accuracies)),
        "all_seeds_reach_overfit_threshold": all(value >= OVERFIT_THRESHOLD for value in best_accuracies),
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    matched = bool(summaries["matched_supervised"]["all_seeds_reach_overfit_threshold"])
    ce_only = bool(summaries["lam_ce_only"]["all_seeds_reach_overfit_threshold"])
    full = bool(summaries["lam_full_objective"]["all_seeds_reach_overfit_threshold"])
    if matched and ce_only and not full:
        return "AUXILIARY_OBJECTIVE_INTERFERENCE"
    if matched and not ce_only:
        return "LAM_BACKBONE_OR_LATENT_PATH_TRAINABILITY_FAILURE"
    if not matched:
        return "SHARED_ENCODER_OR_OPTIMIZATION_TRAINABILITY_FAILURE"
    if matched and ce_only and full:
        return "TRAINABILITY_PASSES_COLLAPSE_IS_GENERALIZATION_OR_DATA_OBJECTIVE_INTERACTION"
    return "MIXED_OR_SEED_SENSITIVE_TRAINABILITY_FAILURE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only ARC overfit diagnostic for collapse root-cause isolation.")
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
    subset = balanced_subset(eligible, per_class=args.per_class)
    label_counts = Counter(example.label for example in subset)
    expected_per_class = {label: args.per_class for label in range(4)}
    if dict(label_counts) != expected_per_class:
        raise RuntimeError(f"diagnostic subset is not balanced: {dict(label_counts)}")

    cfg = LAMJEPAConfig()
    tokens, numeric_x, labels = batchify(subset, vocab_size=cfg.vocab_size, device="cpu")
    lam_active, lam_total = probe_lam_active_parameters(subset, cfg=cfg, batch_size=len(subset), model_steps=args.model_steps, device="cpu")
    depth, hidden_dim, matched_total, parameter_gap = choose_matched_architecture(cfg, target_active_parameters=lam_active, tolerance=0.01)

    conditions: dict[str, list[dict[str, object]]] = {"matched_supervised": [], "lam_ce_only": [], "lam_full_objective": []}
    for seed in args.seeds:
        conditions["matched_supervised"].append({"seed": seed, "history": train_matched(cfg, tokens, numeric_x, labels, seed=seed, hidden_dim=hidden_dim, depth=depth, lr=args.learning_rate)})
        conditions["lam_ce_only"].append({"seed": seed, "history": train_lam(cfg, tokens, numeric_x, labels, seed=seed, lr=args.learning_rate, model_steps=args.model_steps, full_objective=False)})
        conditions["lam_full_objective"].append({"seed": seed, "history": train_lam(cfg, tokens, numeric_x, labels, seed=seed, lr=args.learning_rate, model_steps=args.model_steps, full_objective=True)})

    summaries = {name: summarize_condition(records) for name, records in conditions.items()}
    result = {
        "artifact_type": "LAM-JEPA ARC trainability overfit diagnostic",
        "scope": "training-only diagnostic; no validation or test access",
        "protocol_context": "lam-jepa-arc-challenge-v4",
        "diagnostic_rule": "balanced first-N eligible train examples per class; no performance/generalization claim",
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
        "capacity": {
            "lam_gradient_active_parameters": lam_active,
            "lam_total_trainable_parameters": lam_total,
            "matched_trainable_parameters": matched_total,
            "matched_ratio_to_lam_active": matched_total / lam_active,
            "matched_depth": depth,
            "matched_hidden_dim": hidden_dim,
            "matched_parameter_gap": parameter_gap,
        },
        "conditions": conditions,
        "summaries": summaries,
        "diagnosis": classify(summaries),
        "claim_boundary": {
            "validation_accessed": False,
            "test_accessed": False,
            "performance_claim_authorized": False,
            "mechanism_claim_authorized": False,
            "research_complete": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnosis": result["diagnosis"], "summaries": summaries, "capacity": result["capacity"]}, indent=2))


if __name__ == "__main__":
    main()
