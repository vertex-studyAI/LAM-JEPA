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
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.benchmarking.runner import evaluate_tasks
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Evaluate OOD generalization by holding out one task.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--holdout", type=str, default="chain")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg).to(args.device)
    load_checkpoint(args.checkpoint, model, map_location=args.device)
    in_tasks = tuple(t for t in ("parity", "modadd", "algebra", "chain") if t != args.holdout)
    in_res = evaluate_tasks(model, cfg, tasks=in_tasks)
    ood_res = evaluate_tasks(model, cfg, tasks=(args.holdout,))
    out = {"in_domain": in_res, "ood": ood_res}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
