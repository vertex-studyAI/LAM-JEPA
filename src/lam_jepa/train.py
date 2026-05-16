from __future__ import annotations
import argparse
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LAMJEPA, LAMJEPAConfig
from .data import sample_batch, Curriculum
from .losses import total_loss
from .utils import set_seed


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, task: str, vocab_size: int, batch_size: int):
        super().__init__()
        self.task = task
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = sample_batch(self.task, batch=self.batch_size, vocab_size=self.vocab_size)
            yield {
                "tokens": batch.tokens,
                "numeric_x": batch.numeric_x,
                "labels": batch.labels,
                "rubric": batch.rubric,
            }


def train(
    steps: int = 100,
    device: str = "cpu",
    batch_size: int = 16,
    lr: float = 3e-4,
    task: str = "mixed",
    seed: int = 42,
    cfg: LAMJEPAConfig | None = None,
):
    set_seed(seed)
    if cfg is None:
        cfg = LAMJEPAConfig()
    model = LAMJEPA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    curriculum = Curriculum()
    history = []

    for step in tqdm(range(steps)):
        current_task = task if task != "mixed" else curriculum.sample()
        batch = sample_batch(current_task, batch=batch_size, vocab_size=cfg.vocab_size)
        tokens = batch.tokens.to(device)
        numeric_x = batch.numeric_x.to(device)
        labels = batch.labels.to(device)
        rubric = batch.rubric.to(device)

        outputs = model(tokens, numeric_x=numeric_x, steps=0)
        loss, stats = total_loss(outputs, labels, rubric)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_target()

        pred = outputs["logits"].argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        curriculum.update(acc)

        history.append({"step": step, "task": current_task, "loss": stats["total"], "acc": acc, "conf": stats["conf"], **stats})

        if step % 10 == 0:
            print(f"step={step:05d} task={current_task} loss={stats['total']:.4f} acc={acc:.3f} conf={stats['conf']:.4f}")

    model.training_history = history
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="checkpoints/lam_jepa.pt")
    args = parser.parse_args()

    model, cfg = train(steps=args.steps, device=args.device, batch_size=args.batch_size, lr=args.lr)
    out = {"model": model.state_dict(), "config": asdict(cfg)}
    torch.save(out, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
