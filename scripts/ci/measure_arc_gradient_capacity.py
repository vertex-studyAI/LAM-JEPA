from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lam_jepa.benchmarking.arc_challenge import LAMARCClassifier, _lam_arc_loss
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure parameters that are actually connected to the exact ARC training objective."
    )
    parser.add_argument("--report", type=Path, default=Path("ci-evidence/arc-gradient-capacity.json"))
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--model-steps", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 4:
        raise SystemExit("batch-size must be >= 4 so all ARC labels are represented")
    if args.model_steps < 1:
        raise SystemExit("model-steps must be >= 1")

    set_seed(args.seed)
    cfg = LAMJEPAConfig()
    model = LAMARCClassifier(cfg, num_choices=4)
    model.train()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    tokens = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(args.batch_size, args.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    numeric_x = torch.zeros(args.batch_size, 1, dtype=torch.float32)
    labels = torch.tensor([index % 4 for index in range(args.batch_size)], dtype=torch.long)

    model.zero_grad(set_to_none=True)
    logits, outputs = model(
        tokens,
        numeric_x,
        model_steps=args.model_steps,
        deterministic=False,
    )
    loss = _lam_arc_loss(logits, outputs, labels)
    loss.backward()

    active: list[dict[str, object]] = []
    inactive: list[dict[str, object]] = []
    total_parameters = 0
    trainable_parameters = 0
    gradient_active_parameters = 0

    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
        row = {
            "name": name,
            "parameters": count,
            "requires_grad": bool(parameter.requires_grad),
        }
        if parameter.grad is not None:
            gradient_active_parameters += count
            row["grad_l1"] = float(parameter.grad.detach().abs().sum().item())
            active.append(row)
        else:
            inactive.append(row)

    active_names = {row["name"] for row in active}
    inactive_names = {row["name"] for row in inactive}

    required_active_prefixes = (
        "backbone.encoder.",
        "backbone.projector.",
        "backbone.latent_action.",
        "backbone.latent_summary_head.",
        "choice_head.",
    )
    for prefix in required_active_prefixes:
        if not any(name.startswith(prefix) for name in active_names):
            raise SystemExit(f"expected ARC-active parameter prefix missing: {prefix}")

    required_inactive_prefixes = (
        "backbone.target_encoder.",
        "backbone.target_projector.",
        "backbone.decoder.",
        "backbone.value_head.",
        "backbone.confidence_head.",
        "backbone.verifier_head.",
        "backbone.rubric_head.",
        "backbone.uncertainty_head.",
    )
    for prefix in required_inactive_prefixes:
        matching = [name for name in active_names if name.startswith(prefix)]
        if matching:
            raise SystemExit(f"ARC-inactive parameter prefix unexpectedly received gradients: {prefix}: {matching[:3]}")
        if not any(name.startswith(prefix) for name in inactive_names):
            raise SystemExit(f"expected inactive parameter prefix not found in model: {prefix}")

    if gradient_active_parameters <= 0:
        raise SystemExit("ARC gradient-active capacity must be positive")
    if gradient_active_parameters >= trainable_parameters:
        raise SystemExit(
            "accounting guard failed: ARC objective unexpectedly activates every trainable parameter; "
            "the known target/auxiliary-head distinction disappeared"
        )

    report = {
        "status": "passed",
        "accounting_definition": "sum numel(parameter) where parameter.grad is not None after exact _lam_arc_loss backward",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "model_steps": args.model_steps,
        "loss": float(loss.detach().item()),
        "total_parameters": total_parameters,
        "requires_grad_parameters": trainable_parameters,
        "gradient_active_parameters": gradient_active_parameters,
        "gradient_inactive_parameters": trainable_parameters - gradient_active_parameters,
        "gradient_active_fraction_of_requires_grad": gradient_active_parameters / trainable_parameters,
        "active_parameters": active,
        "inactive_parameters": inactive,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "total_parameters",
                    "requires_grad_parameters",
                    "gradient_active_parameters",
                    "gradient_inactive_parameters",
                    "gradient_active_fraction_of_requires_grad",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
