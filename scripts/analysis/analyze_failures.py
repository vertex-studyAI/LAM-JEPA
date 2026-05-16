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
from lam_jepa.data import sample_batch
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Analyze failure cases by confidence.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, default="chain")
    p.add_argument("--out", type=str, default="outputs/failure_analysis.json")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    b = sample_batch(args.task, batch=64, vocab_size=cfg.vocab_size)
    out = model(b.tokens, numeric_x=b.numeric_x, steps=0)
    pred = out["logits"].argmax(dim=-1)
    correct = pred.eq(b.labels)
    conf = out["confidence"].detach().squeeze(-1)
    failures = torch.topk((1.0 - conf).float(), k=min(10, conf.numel())).indices.tolist()
    result = {
        "accuracy": float(correct.float().mean().item()),
        "mean_confidence": float(conf.mean().item()),
        "hardest_indices": failures,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
