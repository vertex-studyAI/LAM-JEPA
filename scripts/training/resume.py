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
from pathlib import Path
import json
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.trainers.trainer import Trainer, TrainerConfig


def main():
    p = argparse.ArgumentParser(description="Resume LAM-JEPA training from checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="experiments/checkpoints/resumed.pt")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(steps=args.steps, device=args.device)
    trainer = Trainer(model, cfg, tcfg)
    trainer.load(args.checkpoint)
    trainer.train_cfg.steps = args.steps
    trainer.fit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trainer.model.state_dict(), "config": cfg.__dict__, "history": trainer.history}, out)
    print(json.dumps({"resumed_from": args.checkpoint, "saved": str(out)}, indent=2))


if __name__ == "__main__":
    main()
