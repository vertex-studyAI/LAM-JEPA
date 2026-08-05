from __future__ import annotations

import argparse
import json
import shutil
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
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for canonical resumable checkpoints (README-compatible).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Legacy alias for --out-dir.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional copy of the canonical final checkpoint at another path.",
    )
    args = parser.parse_args()

    if args.out_dir and args.checkpoint_dir:
        parser.error("use only one of --out-dir or --checkpoint-dir")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    checkpoint_dir = Path(args.out_dir or args.checkpoint_dir or "experiments/checkpoints")
    cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg)
    tcfg = TrainerConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        task=args.task,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        eval_every=max(args.steps // 4, 1),
        save_every=max(args.steps // 2, 1),
    )
    trainer = Trainer(model, cfg, tcfg)
    trainer.fit()

    final_checkpoint = checkpoint_dir / "final.pt"
    if not final_checkpoint.is_file():
        raise RuntimeError(f"trainer did not create the canonical checkpoint: {final_checkpoint}")

    exported_checkpoint = None
    if args.out:
        export_path = Path(args.out)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.resolve() != final_checkpoint.resolve():
            shutil.copy2(final_checkpoint, export_path)
        exported_checkpoint = str(export_path)

    print(
        json.dumps(
            {
                "saved": str(final_checkpoint),
                "exported": exported_checkpoint,
                "final_step": trainer.step,
                "last": trainer.history[-1] if trainer.history else {},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
