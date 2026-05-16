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
import numpy as np
import matplotlib.pyplot as plt
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.data import sample_batch
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Plot latent rollout similarity.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="modadd")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--out", type=str, default="outputs/figures/rollout_similarity.png")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    b = sample_batch(args.task, batch=8, vocab_size=cfg.vocab_size)
    out = model(b.tokens, numeric_x=b.numeric_x, steps=args.steps)
    sims = []
    for z in out["traj"]:
        z = torch.nn.functional.normalize(z.detach().cpu(), dim=-1)
        sims.append((z @ z.t()).numpy())
    sim = np.mean(np.stack(sims, axis=0), axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Latent self-similarity")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
