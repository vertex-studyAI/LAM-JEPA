from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def values_equal(left: Any, right: Any) -> bool:
    """Compare checkpoint state by semantic value, not serialization identity."""
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(values_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def compare_tensors(left: dict[str, Any], right: dict[str, Any], namespace: str) -> list[str]:
    require(set(left) == set(right), f"{namespace} keys differ")
    mismatches: list[str] = []
    for key in sorted(left):
        if not values_equal(left[key], right[key]):
            mismatches.append(key)
    return mismatches


def normalized_extra(payload: dict[str, Any]) -> dict[str, Any]:
    extra = dict(payload.get("extra") or {})
    train_config = dict(extra.get("train_config") or {})
    # Output locations are deliberately different between replay runs. They do
    # not affect model/data semantics and therefore are excluded from equality.
    train_config.pop("checkpoint_dir", None)
    train_config.pop("log_dir", None)
    extra["train_config"] = train_config
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that two same-seed train_single runs replay exactly."
    )
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.first.is_file(), f"checkpoint not found: {args.first}")
    require(args.second.is_file(), f"checkpoint not found: {args.second}")

    first = torch.load(args.first, map_location="cpu", weights_only=False)
    second = torch.load(args.second, map_location="cpu", weights_only=False)
    require(isinstance(first, dict) and isinstance(second, dict), "checkpoints must be dictionaries")

    model_mismatches = compare_tensors(first.get("model", {}), second.get("model", {}), "model")
    require(not model_mismatches, f"model replay mismatch: {model_mismatches[:8]}")

    require(values_equal(first.get("step"), second.get("step")), "checkpoint step differs")
    require(values_equal(first.get("metrics"), second.get("metrics")), "checkpoint metrics differ")
    require(values_equal(normalized_extra(first), normalized_extra(second)), "semantic checkpoint metadata differs")

    first_rng = first.get("rng") or {}
    second_rng = second.get("rng") or {}
    require(values_equal(first_rng, second_rng), "RNG replay state differs")

    metrics = first.get("metrics") or {}
    report = {
        "status": "passed",
        "first": str(args.first),
        "second": str(args.second),
        "seed": ((first.get("extra") or {}).get("train_config") or {}).get("seed"),
        "step": first.get("step"),
        "final_loss": metrics.get("loss"),
        "final_accuracy": metrics.get("acc"),
        "model_tensor_count": len(first.get("model") or {}),
        "model_state_exact": True,
        "metrics_exact": True,
        "semantic_metadata_exact": True,
        "rng_state_exact": True,
        "excluded_nonsemantic_fields": ["checkpoint_dir", "log_dir"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
