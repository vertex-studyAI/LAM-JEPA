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


def main():
    p = argparse.ArgumentParser(description="Check for NaNs in a forward pass.")
    p.add_argument("--task", type=str, default="modadd")
    args = p.parse_args()
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    batch = sample_batch(args.task, batch=8, vocab_size=cfg.vocab_size)
    out = model(batch.tokens, numeric_x=batch.numeric_x, steps=3)
    has_nan = any(torch.isnan(v).any().item() for v in [out["logits"], out["confidence"], out["verifier"], out["rubric"]])
    print({"has_nan": has_nan})


if __name__ == "__main__":
    main()
