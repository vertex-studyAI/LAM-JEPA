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
import numpy as np
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch
from lam_jepa.callbacks.checkpointing.load import load_checkpoint
from lam_jepa.metrics.representation.collapse import collapse_score


def main():
    p = argparse.ArgumentParser(description="Analyze latent geometry.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="modadd")
    p.add_argument("--out", type=str, default="outputs/latent_analysis.json")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    b = sample_batch(args.task, batch=32, vocab_size=cfg.vocab_size)
    out = model(b.tokens, numeric_x=b.numeric_x, steps=3)
    z = out["z_q"].detach().cpu()
    result = {
        "collapse_score": collapse_score(z),
        "latent_norm": float(z.norm(dim=-1).mean().item()),
        "trajectory_length": len(out["traj"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
