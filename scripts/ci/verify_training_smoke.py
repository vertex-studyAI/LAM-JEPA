from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the checkpoint emitted by the deterministic CI smoke run."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.checkpoint.is_file(), f"checkpoint not found: {args.checkpoint}")

    # This file is generated in the same trusted CI job immediately before it is read.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "checkpoint payload must be a dictionary")

    state = payload.get("model")
    config = payload.get("config")
    history = payload.get("history")
    require(isinstance(state, dict) and state, "checkpoint model state is missing or empty")
    require(isinstance(config, dict) and config, "checkpoint config is missing or empty")
    require(isinstance(history, list) and history, "checkpoint training history is missing or empty")

    tensors = [value for value in state.values() if torch.is_tensor(value)]
    require(tensors, "checkpoint contains no model tensors")
    require(
        all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
        "checkpoint contains non-finite model values",
    )

    last = history[-1]
    require(isinstance(last, dict), "final history entry must be a dictionary")
    for metric in ("loss", "acc"):
        value = last.get(metric)
        require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"final history metric {metric!r} is missing or non-finite",
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "torch_version": torch.__version__,
        "history_entries": len(history),
        "state_tensors": len(tensors),
        "parameters": sum(tensor.numel() for tensor in tensors),
        "final": {
            "step": last.get("step"),
            "task": last.get("task"),
            "loss": float(last["loss"]),
            "accuracy": float(last["acc"]),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
