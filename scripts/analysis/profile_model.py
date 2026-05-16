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
import time
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch


def main():
    p = argparse.ArgumentParser(description="Rudimentary model profile.")
    p.add_argument("--task", type=str, default="modadd")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg).eval()
    b = sample_batch(args.task, batch=32, vocab_size=cfg.vocab_size)
    start = time.time()
    for _ in range(args.iters):
        with torch.no_grad():
            _ = model(b.tokens, numeric_x=b.numeric_x, steps=0)
    elapsed = time.time() - start
    print({"iters": args.iters, "seconds": elapsed, "sec_per_iter": elapsed / max(args.iters, 1)})


if __name__ == "__main__":
    main()
