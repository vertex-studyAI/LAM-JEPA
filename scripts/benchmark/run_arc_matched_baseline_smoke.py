from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from lam_jepa.benchmarking.arc_challenge import (
    LAMARCClassifier,
    _lam_arc_loss,
    _predict_baseline,
    _predict_lam,
    _train_lam_jepa,
    batchify,
    dataset_digest,
    id_digest,
    iter_minibatches,
    load_arc_split,
    score_predictions,
)
from lam_jepa.benchmarking.arc_matched_baseline import (
    MatchedCapacityARCClassifier,
    build_matched_capacity_arc_classifier,
    gradient_active_parameter_count,
    trainable_parameter_count,
)
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed


def measure_lam_arc_gradient_capacity(
    train: Sequence,
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[int, list[str], list[str], int]:
    set_seed(seed)
    model = LAMARCClassifier(cfg, num_choices=4).to(device)
    model.train()
    batch = next(iter(iter_minibatches(train, batch_size, seed)))
    tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
    model.zero_grad(set_to_none=True)
    logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=False)
    loss = _lam_arc_loss(logits, outputs, labels)
    loss.backward()

    active_count = 0
    active_names: list[str] = []
    inactive_names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            inactive_names.append(name)
        else:
            active_names.append(name)
            active_count += int(parameter.numel())
    return active_count, active_names, inactive_names, trainable_parameter_count(model)


def train_matched_baseline(
    train: Sequence,
    *,
    cfg: LAMJEPAConfig,
    target_parameters: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[MatchedCapacityARCClassifier, dict[str, object]]:
    set_seed(seed)
    model, spec = build_matched_capacity_arc_classifier(
        cfg,
        target_parameters=target_parameters,
        num_choices=4,
        allowed_ratio_min=0.99,
        allowed_ratio_max=1.01,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens, numeric_x), labels)
            loss.backward()
            optimizer.step()

    # Re-run one supervised backward pass solely to prove every counted baseline parameter
    # is connected to the actual ARC answer loss.
    model.zero_grad(set_to_none=True)
    proof_batch = next(iter(iter_minibatches(train, batch_size, seed)))
    tokens, numeric_x, labels = batchify(proof_batch, vocab_size=cfg.vocab_size, device=device)
    proof_loss = F.cross_entropy(model(tokens, numeric_x), labels)
    proof_loss.backward()
    active_count, active_names, inactive_names = gradient_active_parameter_count(model)
    trainable_count = trainable_parameter_count(model)
    if inactive_names:
        raise RuntimeError(f"matched baseline has ARC-inactive trainable parameters: {inactive_names[:8]}")
    if active_count != trainable_count:
        raise RuntimeError(
            f"matched baseline active/trainable mismatch: active={active_count}, trainable={trainable_count}"
        )
    ratio = active_count / target_parameters
    if not (0.99 <= ratio <= 1.01):
        raise RuntimeError(
            f"matched baseline parameter ratio outside frozen protocol: {ratio:.6f} "
            f"({active_count}/{target_parameters})"
        )
    model.zero_grad(set_to_none=True)
    model.eval()
    return model, {
        "target_parameters": target_parameters,
        "trainable_parameters": trainable_count,
        "gradient_active_parameters": active_count,
        "parameter_ratio": ratio,
        "hidden_width": spec.hidden_width,
        "proof_loss": float(proof_loss.detach().item()),
        "active_parameter_names": active_names,
        "inactive_parameter_names": inactive_names,
        "module_types": sorted({type(module).__name__ for module in model.modules()}),
    }


def run(
    train_path: Path,
    validation_path: Path,
    *,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    model_steps: int,
    train_limit: int | None,
    validation_limit: int | None,
    device: str,
) -> dict[str, object]:
    train_all = load_arc_split(train_path)
    validation_all = load_arc_split(validation_path)
    train = list(train_all[:train_limit] if train_limit else train_all)
    validation = list(validation_all[:validation_limit] if validation_limit else validation_all)
    if not train or not validation:
        raise ValueError("ARC train and validation data must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        raise ValueError("matched ARC smoke requires exactly four choices")
    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    if overlap:
        raise ValueError(f"ARC train/validation ID leakage detected: {overlap[:5]}")

    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique and non-empty")

    cfg = LAMJEPAConfig()
    expected_ids = [example.item_id for example in validation]
    records: list[dict[str, object]] = []

    for seed in normalized_seeds:
        target, lam_active_names, lam_inactive_names, lam_trainable = measure_lam_arc_gradient_capacity(
            train,
            cfg=cfg,
            seed=seed,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        matched, matched_accounting = train_matched_baseline(
            train,
            cfg=cfg,
            target_parameters=target,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_steps=model_steps,
            device=device,
        )

        matched_probs, labels, matched_rows = _predict_baseline(
            matched,
            validation,
            cfg=cfg,
            batch_size=batch_size,
            device=device,
        )
        lam_probs, lam_labels, lam_rows = _predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        if not torch.equal(labels, lam_labels):
            raise RuntimeError("matched baseline and LAM-JEPA labels are not aligned")
        if [row["id"] for row in matched_rows] != expected_ids:
            raise RuntimeError("matched baseline validation row order changed")
        if [row["id"] for row in lam_rows] != expected_ids:
            raise RuntimeError("LAM-JEPA validation row order changed")

        records.append(
            {
                "seed": seed,
                "lam_parameter_accounting": {
                    "requires_grad_parameters": lam_trainable,
                    "gradient_active_parameters": target,
                    "gradient_inactive_parameters": lam_trainable - target,
                    "active_parameter_names": lam_active_names,
                    "inactive_parameter_names": lam_inactive_names,
                },
                "matched_baseline_parameter_accounting": matched_accounting,
                "matched_baseline": score_predictions(matched_probs, labels),
                "lam_jepa": score_predictions(lam_probs, lam_labels),
                "raw_predictions": {
                    "matched_baseline": matched_rows,
                    "lam_jepa": lam_rows,
                },
            }
        )

    return {
        "protocol": {
            "protocol_id": "lam-jepa-arc-challenge-v2",
            "dataset": "AI2 ARC-Challenge",
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
            "seeds": normalized_seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_steps": model_steps,
            "train_examples": len(train),
            "validation_examples": len(validation),
            "capacity_accounting": "gradient-active parameters after exact ARC loss backward",
            "allowed_parameter_ratio": [0.99, 1.01],
            "test_split_policy": "not loaded or accessed by this smoke",
            "claim_boundary": "Development smoke only. No performance, superiority, test-set, or research-complete claim is permitted from this output."
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the protocol-v2 matched-capacity ARC development smoke.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = run(
        args.train,
        args.validation,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_steps=args.model_steps,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "protocol": result["protocol"],
        "records": [
            {
                "seed": record["seed"],
                "lam_gradient_active_parameters": record["lam_parameter_accounting"]["gradient_active_parameters"],
                "matched_gradient_active_parameters": record["matched_baseline_parameter_accounting"]["gradient_active_parameters"],
                "parameter_ratio": record["matched_baseline_parameter_accounting"]["parameter_ratio"],
                "hidden_width": record["matched_baseline_parameter_accounting"]["hidden_width"],
                "matched_accuracy": record["matched_baseline"]["accuracy"],
                "lam_accuracy": record["lam_jepa"]["accuracy"],
            }
            for record in result["records"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
