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
import torch

from lam_jepa.model import LAMJEPA, LAMJEPAConfig
from lam_jepa.callbacks.checkpointing.load import load_checkpoint


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: LAMJEPA):
        super().__init__()
        self.model = model

    def forward(self, tokens, numeric_x):
        out = self.model(tokens, numeric_x=numeric_x, steps=0)
        return out["logits"], out["confidence"], out["verifier"]


def main():
    p = argparse.ArgumentParser(description="Export model to ONNX.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/lam_jepa.onnx")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = LAMJEPAConfig(**ckpt.get("config", {}))
    model = LAMJEPA(cfg)
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.eval()
    wrapper = ExportWrapper(model)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    numeric_x = torch.randn(2, cfg.input_dim)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (tokens, numeric_x), args.out, input_names=["tokens", "numeric_x"], output_names=["logits", "confidence", "verifier"], opset_version=17)
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
