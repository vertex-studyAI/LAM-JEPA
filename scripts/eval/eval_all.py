from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS, evaluate_model
from lam_jepa.callbacks.checkpointing.load import load_checkpoint
from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a canonical checkpoint on all ed-tech benchmarks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval_all.json"))
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.batches < 1:
        parser.error("--batches must be at least 1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    set_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg_payload = ckpt.get("extra", {}).get("config", ckpt.get("config", {}))
    if not isinstance(cfg_payload, dict) or not cfg_payload:
        parser.error("checkpoint does not contain a model configuration")

    cfg = LAMJEPAConfig(**cfg_payload)
    model = LAMJEPA(cfg).to(args.device)
    loaded = load_checkpoint(args.checkpoint, model, map_location=args.device)

    # Model construction and checkpoint loading may consume RNG state. Reset the
    # benchmark sampler immediately before evaluation so a separately executed
    # baseline command with the same protocol can reproduce the exact rows.
    set_seed(args.seed)
    scores = evaluate_model(
        model,
        cfg,
        tasks=EDTECH_TASKS,
        batch_size=args.batch_size,
        batches=args.batches,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": loaded.get("step"),
        "device": str(next(model.parameters()).device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "scores": scores,
        "tasks": list(EDTECH_TASKS),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
