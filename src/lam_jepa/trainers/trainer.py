from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import autocast, GradScaler

from ..data import Curriculum, sample_batch
from ..losses import total_loss
from ..model import LAMJEPA, LAMJEPAConfig
from ..utils import set_seed
from ..callbacks.checkpointing.save import save_checkpoint
from ..callbacks.checkpointing.load import load_checkpoint
from ..callbacks.early_stop import EarlyStopper


@dataclass
class TrainerConfig:
    steps: int = 1000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    task: str = "mixed"
    seed: int = 42
    amp: bool = False
    device: str = "cpu"
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"
    eval_every: int = 100
    save_every: int = 200
    early_stop_patience: int = 0


class Trainer:
    def __init__(self, model: LAMJEPA, cfg: LAMJEPAConfig, train_cfg: TrainerConfig):
        set_seed(train_cfg.seed)
        self.model = model
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(train_cfg.steps, 1))
        self.scaler = GradScaler(enabled=train_cfg.amp and self.device.type == "cuda")
        self.curriculum = Curriculum()
        self.history = []
        self.step = 0
        self.early_stopper = EarlyStopper(patience=train_cfg.early_stop_patience) if train_cfg.early_stop_patience > 0 else None

    def sample_task(self) -> str:
        return self.train_cfg.task if self.train_cfg.task != "mixed" else self.curriculum.sample()

    def batch(self):
        task = self.sample_task()
        return task, sample_batch(task, batch=self.train_cfg.batch_size, vocab_size=self.cfg.vocab_size)

    def train_step(self) -> Dict[str, float]:
        self.model.train()
        task, batch = self.batch()
        tokens = batch.tokens.to(self.device)
        numeric_x = batch.numeric_x.to(self.device)
        labels = batch.labels.to(self.device)
        rubric = batch.rubric.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=self.scaler.is_enabled()):
            outputs = self.model(tokens, numeric_x=numeric_x, steps=0)
            loss, stats = total_loss(outputs, labels, rubric)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.opt.step()

        self.scheduler.step()
        self.model.update_target()

        pred = outputs["logits"].argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        self.curriculum.update(acc)

        record = {"step": self.step, "task": task, "acc": acc, **stats}
        self.history.append(record)
        self.step += 1
        return record

    @torch.no_grad()
    def evaluate(self, task: str, batches: int = 8) -> Dict[str, float]:
        self.model.eval()
        preds = []
        labels = []
        confs = []
        rubrics = []
        for _ in range(batches):
            batch = sample_batch(task, batch=self.train_cfg.batch_size, vocab_size=self.cfg.vocab_size)
            out = self.model(batch.tokens.to(self.device), numeric_x=batch.numeric_x.to(self.device), steps=0)
            preds.append(out["logits"].argmax(dim=-1).cpu())
            labels.append(batch.labels.cpu())
            confs.append(out["confidence"].detach().cpu())
            rubrics.append(out["rubric"].detach().cpu())
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        conf = torch.cat(confs).mean().item()
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc, "confidence": conf}

    def fit(self) -> LAMJEPA:
        stopper_triggered = False
        for _ in range(self.train_cfg.steps):
            rec = self.train_step()
            if self.train_cfg.eval_every and (self.step % self.train_cfg.eval_every == 0):
                _ = self.evaluate("parity", batches=2)
            if self.early_stopper is not None and self.early_stopper.step(rec["acc"]):
                stopper_triggered = True
                break
            if self.train_cfg.save_every and (self.step % self.train_cfg.save_every == 0):
                self.save("latest.pt")
        self.save("final.pt")
        self.model.eval()
        self.stopped_early = stopper_triggered
        return self.model

    def save(self, name: str = "latest.pt", extra: Optional[Dict] = None) -> Path:
        ckpt_dir = Path(self.train_cfg.checkpoint_dir)
        path = ckpt_dir / name
        return save_checkpoint(
            path,
            self.model,
            optimizer=self.opt,
            scheduler=self.scheduler,
            step=self.step,
            metrics=self.history[-1] if self.history else {},
            extra={"config": asdict(self.cfg), "train_config": asdict(self.train_cfg), **(extra or {})},
        )

    def load(self, path: str | Path):
        ckpt = load_checkpoint(path, self.model, self.opt, self.scheduler, map_location=self.device)
        self.step = int(ckpt.get("step", self.step))
        self.history = list(ckpt.get("metrics_history", self.history)) if isinstance(ckpt.get("metrics_history"), list) else self.history
        return ckpt
