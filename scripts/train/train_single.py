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

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.training import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LAM-JEPA with the research trainer.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--task", type=str, default="mixed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--out", type=str, default="experiments/checkpoints/final.pt")
    args = parser.parse_args()

    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, task=args.task, seed=args.seed, device=args.device, checkpoint_dir=args.checkpoint_dir, eval_every=max(args.steps // 4, 1), save_every=max(args.steps // 2, 1))
    trainer = Trainer(model, cfg, tcfg)
    trained = trainer.fit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trained.state_dict(), "config": cfg.__dict__, "history": trainer.history}, out)
    print(json.dumps({"saved": str(out), "final_step": trainer.step, "last": trainer.history[-1] if trainer.history else {}}, indent=2))


if __name__ == "__main__":
    main()
