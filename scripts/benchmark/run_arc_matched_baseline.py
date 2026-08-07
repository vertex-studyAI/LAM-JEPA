from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    LAMARCClassifier,
    _lam_arc_loss,
    _train_lam_jepa,
    batchify,
    dataset_digest,
    id_digest,
    iter_minibatches,
    load_arc_split,
    reverse_choices,
    score_predictions,
)
from lam_jepa.model import LAMJEPAConfig, MultiViewEncoder
from lam_jepa.utils import set_seed


class ResidualSupervisedBlock(nn.Module):
    """Trainable residual MLP block used only by the supervised comparison model."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))
        return x + update


class MatchedSupervisedClassifier(nn.Module):
    """Supervised encoder with active trainable capacity matched to LAM-JEPA."""

    def __init__(self, cfg: LAMJEPAConfig, *, hidden_dim: int, depth: int, num_choices: int = 4):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.blocks = nn.ModuleList(
            [ResidualSupervisedBlock(cfg.proj_dim, hidden_dim, cfg.dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(cfg.proj_dim)
        self.classifier = nn.Linear(cfg.proj_dim, num_choices)

    def forward(self, tokens: torch.Tensor, numeric_x: torch.Tensor) -> torch.Tensor:
        x = self.projector(self.encoder(tokens, numeric_x=numeric_x))
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.norm(x))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def gradient_active_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


def probe_lam_active_parameters(
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[int, int]:
    if not examples:
        raise ValueError("cannot probe LAM-JEPA parameter activity on an empty dataset")
    set_seed(0)
    model = LAMARCClassifier(cfg, num_choices=4).to(device)
    model.train()
    probe = list(examples[: min(len(examples), batch_size)])
    tokens, numeric_x, labels = batchify(probe, vocab_size=cfg.vocab_size, device=device)
    model.zero_grad(set_to_none=True)
    logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=False)
    loss = _lam_arc_loss(logits, outputs, labels)
    loss.backward()
    active = gradient_active_parameter_count(model)
    total = trainable_parameter_count(model)
    if active <= 0 or active > total:
        raise RuntimeError(f"invalid LAM-JEPA active parameter count: active={active} total={total}")
    return active, total


def choose_matched_architecture(
    cfg: LAMJEPAConfig,
    *,
    target_active_parameters: int,
    tolerance: float,
) -> tuple[int, int, int, float]:
    if target_active_parameters <= 0:
        raise ValueError("target_active_parameters must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between 0 and 1")

    best: tuple[float, int, int, int] | None = None
    for depth in range(1, 9):
        for hidden_dim in range(16, 1025, 16):
            model = MatchedSupervisedClassifier(
                cfg,
                hidden_dim=hidden_dim,
                depth=depth,
                num_choices=4,
            )
            count = trainable_parameter_count(model)
            gap = abs(count - target_active_parameters) / target_active_parameters
            candidate = (gap, depth, hidden_dim, count)
            if best is None or candidate < best:
                best = candidate

    assert best is not None
    gap, depth, hidden_dim, count = best
    if gap > tolerance:
        raise RuntimeError(
            "could not construct an active supervised baseline within the requested parameter "
            f"tolerance: target={target_active_parameters} best={count} gap={gap:.6f}"
        )
    return depth, hidden_dim, count, gap


def train_matched_baseline(
    train: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    hidden_dim: int,
    depth: int,
) -> tuple[MatchedSupervisedClassifier, int]:
    set_seed(seed)
    model = MatchedSupervisedClassifier(
        cfg,
        hidden_dim=hidden_dim,
        depth=depth,
        num_choices=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    observed_active: int | None = None

    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens, numeric_x), labels)
            loss.backward()
            if observed_active is None:
                observed_active = gradient_active_parameter_count(model)
            optimizer.step()

    if observed_active is None:
        raise RuntimeError("matched baseline did not execute a training step")
    total = trainable_parameter_count(model)
    if observed_active != total:
        raise RuntimeError(
            "matched baseline contains trainable parameters that did not receive gradients: "
            f"active={observed_active} total={total}"
        )
    return model.eval(), observed_active


@torch.no_grad()
def predict_matched(
    model: MatchedSupervisedClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        batch_probabilities = torch.softmax(model(tokens, numeric_x), dim=-1).cpu()
        probabilities.append(batch_probabilities)
        labels.append(y.cpu())
        for example, probability, label in zip(batch, batch_probabilities, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )

    return torch.cat(probabilities), torch.cat(labels), rows


@torch.no_grad()
def predict_lam(
    model: LAMARCClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
        if len(outputs["actions"]) != model_steps:
            raise RuntimeError(
                f"LAM-JEPA evaluation expected {model_steps} planner steps, got {len(outputs['actions'])}"
            )
        batch_probabilities = torch.softmax(logits, dim=-1).cpu()
        probabilities.append(batch_probabilities)
        labels.append(y.cpu())
        for example, probability, label in zip(batch, batch_probabilities, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )

    return torch.cat(probabilities), torch.cat(labels), rows


def mean_std(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LAM-JEPA with a gradient-active parameter-matched supervised ARC baseline."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--match-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.model_steps < 1:
        parser.error("--epochs, --batch-size, and --model-steps must be at least 1")

    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two unique values")

    full_train = load_arc_split(args.train)
    full_validation = load_arc_split(args.validation)
    train = list(full_train[: args.train_limit] if args.train_limit else full_train)
    validation = list(full_validation[: args.validation_limit] if args.validation_limit else full_validation)
    if not train or not validation:
        parser.error("train and validation splits must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        parser.error("the matched ARC protocol currently requires exactly four choices")

    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    if overlap:
        raise SystemExit(f"train/validation leakage detected: {overlap[:5]}")

    cfg = LAMJEPAConfig()
    lam_active, lam_total = probe_lam_active_parameters(
        train,
        cfg=cfg,
        batch_size=args.batch_size,
        model_steps=args.model_steps,
        device=args.device,
    )
    depth, hidden_dim, baseline_total, parameter_gap = choose_matched_architecture(
        cfg,
        target_active_parameters=lam_active,
        tolerance=args.match_tolerance,
    )

    reversed_validation = [reverse_choices(example) for example in validation]
    records: list[dict[str, object]] = []
    lam_accuracies: list[float] = []
    matched_accuracies: list[float] = []
    paired_deltas: list[float] = []

    for seed in seeds:
        matched, baseline_active = train_matched_baseline(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            device=args.device,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        if baseline_active != baseline_total:
            raise RuntimeError(
                f"baseline parameter activity changed: expected {baseline_total}, observed {baseline_active}"
            )

        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_steps=args.model_steps,
            device=args.device,
        )

        lam_probs, lam_labels, lam_rows = predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        matched_probs, matched_labels, matched_rows = predict_matched(
            matched,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            device=args.device,
        )
        if not torch.equal(lam_labels, matched_labels):
            raise RuntimeError("LAM-JEPA and matched baseline evaluated different validation labels")

        lam_reverse_probs, lam_reverse_labels, lam_reverse_rows = predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        matched_reverse_probs, matched_reverse_labels, matched_reverse_rows = predict_matched(
            matched,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            device=args.device,
        )
        if not torch.equal(lam_reverse_labels, matched_reverse_labels):
            raise RuntimeError("LAM-JEPA and matched baseline evaluated different reversed labels")

        lam_metrics = score_predictions(lam_probs, lam_labels)
        matched_metrics = score_predictions(matched_probs, matched_labels)
        lam_reverse_metrics = score_predictions(lam_reverse_probs, lam_reverse_labels)
        matched_reverse_metrics = score_predictions(matched_reverse_probs, matched_reverse_labels)
        delta = float(lam_metrics["accuracy"] - matched_metrics["accuracy"])
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        matched_accuracies.append(float(matched_metrics["accuracy"]))
        paired_deltas.append(delta)

        records.append(
            {
                "seed": seed,
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_reverse_metrics,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_reverse_rows,
                },
                "matched_supervised": {
                    "metrics": matched_metrics,
                    "choice_reversal_metrics": matched_reverse_metrics,
                    "predictions": matched_rows,
                    "choice_reversal_predictions": matched_reverse_rows,
                },
                "accuracy_delta_lam_minus_matched": delta,
            }
        )

    payload = {
        "protocol": {
            "dataset": "AI2 ARC-Challenge",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_steps": args.model_steps,
            "device": args.device,
            "primary_metric": "multiple-choice accuracy",
            "calibration_metrics": ["brier", "ece", "mean_true_class_probability"],
            "robustness_check": "deterministic reversal of answer-choice order with label remapping",
            "parameter_match_basis": (
                "gradient-active trainable parameters under each ARC supervised objective; "
                "EMA target or other parameters without gradients are excluded from the LAM-JEPA match target"
            ),
            "lam_total_trainable_parameters": lam_total,
            "lam_gradient_active_parameters": lam_active,
            "matched_supervised_trainable_parameters": baseline_total,
            "matched_supervised_gradient_active_parameters": baseline_total,
            "matched_supervised_depth": depth,
            "matched_supervised_hidden_dim": hidden_dim,
            "parameter_relative_gap": parameter_gap,
            "parameter_match_tolerance": args.match_tolerance,
            "strong_pretrained_baseline": "NOT_INCLUDED",
            "final_seed_requirement": 5,
            "test_split_policy": "not downloaded or evaluated by this development command",
            "claim_boundary": (
                "This comparison closes only the capacity-matched supervised-baseline plumbing gap. "
                "The smoke is not a strong pretrained comparison, not a locked-test result, not independent "
                "reproduction, and not evidence that LAM-JEPA is superior or RESEARCH_COMPLETE."
            ),
        },
        "records": records,
        "summary": {
            "lam_accuracy": mean_std(lam_accuracies),
            "matched_supervised_accuracy": mean_std(matched_accuracies),
            "paired_accuracy_delta_lam_minus_matched": mean_std(paired_deltas),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
