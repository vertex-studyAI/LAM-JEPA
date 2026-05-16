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

import matplotlib.pyplot as plt
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


def main():
    p = argparse.ArgumentParser(description="Plot training history from a checkpoint or JSON file.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/figures/training_curve.png")
    args = p.parse_args()
    path = Path(args.input)
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        history = ckpt.get("history", [])
    else:
        history = json.loads(path.read_text())
    if not history:
        raise SystemExit("No history found.")
    steps = [h.get("step", i) for i, h in enumerate(history)]
    loss = [h.get("total", h.get("loss", 0.0)) for h in history]
    acc = [h.get("acc", 0.0) for h in history]
    conf = [h.get("conf", 0.0) for h in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, loss, label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(steps, acc, label="accuracy")
    ax2.plot(steps, conf, label="confidence")
    ax2.set_ylabel("accuracy / confidence")
    plt.title("LAM-JEPA training curve")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
