from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def compare_tensors(left: dict[str, Any], right: dict[str, Any], namespace: str) -> list[str]:
    require(set(left) == set(right), f"{namespace} keys differ")
    mismatches: list[str] = []
    for key in sorted(left):
        a = left[key]
        b = right[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a, b):
                mismatches.append(key)
        elif a != b:
            mismatches.append(key)
    return mismatches


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

    for field in ("step", "metrics", "extra"):
        require(first.get(field) == second.get(field), f"checkpoint field differs: {field}")

    first_rng = first.get("rng") or {}
    second_rng = second.get("rng") or {}
    require(set(first_rng) == set(second_rng), "RNG state keys differ")
    rng_mismatches: list[str] = []
    for key in sorted(first_rng):
        a = first_rng[key]
        b = second_rng[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a, b):
                rng_mismatches.append(key)
        elif a != b:
            rng_mismatches.append(key)
    require(not rng_mismatches, f"RNG replay mismatch: {rng_mismatches}")

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
        "rng_state_exact": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
