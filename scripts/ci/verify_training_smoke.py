from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lam_jepa.callbacks.checkpointing.load import load_checkpoint
from lam_jepa.model import LAMJEPA, LAMJEPAConfig


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the canonical checkpoint emitted by the deterministic CI smoke run."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.checkpoint.is_file(), f"checkpoint not found: {args.checkpoint}")

    # This file is generated in the same trusted CI job immediately before it is read.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "checkpoint payload must be a dictionary")

    state = payload.get("model")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    rng = payload.get("rng")
    metrics = payload.get("metrics")
    extra = payload.get("extra")
    step = payload.get("step")

    require(isinstance(state, dict) and state, "checkpoint model state is missing or empty")
    require(isinstance(optimizer, dict) and optimizer, "checkpoint optimizer state is missing or empty")
    require(isinstance(scheduler, dict), "checkpoint scheduler state is missing")
    require(isinstance(rng, dict) and rng, "checkpoint RNG state is missing or empty")
    require(isinstance(metrics, dict) and metrics, "checkpoint metrics are missing or empty")
    require(isinstance(extra, dict), "checkpoint extra metadata is missing")
    require(isinstance(step, int) and step >= 1, "checkpoint step must be a positive integer")

    config = extra.get("config")
    train_config = extra.get("train_config")
    require(isinstance(config, dict) and config, "model configuration is missing or empty")
    require(isinstance(train_config, dict) and train_config, "training configuration is missing or empty")

    tensors = [value for value in state.values() if torch.is_tensor(value)]
    require(tensors, "checkpoint contains no model tensors")
    require(
        all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
        "checkpoint contains non-finite model values",
    )

    for metric in ("loss", "acc"):
        value = metrics.get(metric)
        require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"final metric {metric!r} is missing or non-finite",
        )

    cfg = LAMJEPAConfig(**config)
    reloaded_model = LAMJEPA(cfg)
    loaded = load_checkpoint(args.checkpoint, reloaded_model, map_location="cpu")
    require(loaded.get("step") == step, "checkpoint API returned an inconsistent step")

    report = {
        "checkpoint": str(args.checkpoint),
        "torch_version": torch.__version__,
        "checkpoint_step": step,
        "state_tensors": len(tensors),
        "parameters": sum(tensor.numel() for tensor in tensors),
        "resumable_state": {
            "optimizer": True,
            "scheduler": True,
            "rng": True,
            "model_config": True,
            "training_config": True,
            "repository_loader": True,
        },
        "final": {
            "task": metrics.get("task"),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["acc"]),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
