from __future__ import annotations
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

import argparse
import json
from pathlib import Path
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.benchmarking.runner import evaluate_tasks
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the core ed-tech reasoning tasks.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batches", type=int, default=8)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    results = evaluate_tasks(model, cfg, batches=args.batches)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
