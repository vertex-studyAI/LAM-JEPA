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
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch
from lam_jepa.losses import total_loss


def main():
    p = argparse.ArgumentParser(description="Check for finite gradients.")
    p.add_argument("--task", type=str, default="modadd")
    args = p.parse_args()
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    batch = sample_batch(args.task, batch=8, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=0)
    loss, stats = total_loss(out, batch.labels, batch.rubric)
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    print({"finite_grads": finite, "loss": float(loss.item()), "stats": stats})


if __name__ == "__main__":
    main()
