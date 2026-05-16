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
from lam_jepa.callbacks.checkpointing.load import load_checkpoint
from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on all ed-tech benchmarks.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--out", type=str, default="outputs/eval_all.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("extra", {}).get("config", ckpt.get("config", {})))
    model = LAMJEPA(cfg).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    scores = evaluate_model(model, cfg, tasks=EDTECH_TASKS, batch_size=args.batch_size, batches=args.batches)
    payload = {"checkpoint": args.checkpoint, "scores": scores, "tasks": list(EDTECH_TASKS)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
