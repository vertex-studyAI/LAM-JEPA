from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import json
import math
import os

import torch
from torch.cuda.amp import GradScaler, autocast

from ..callbacks.checkpointing.load import load_checkpoint
from ..callbacks.checkpointing.save import save_checkpoint
from ..data import Curriculum, sample_batch
from ..losses import total_loss
from ..utils import set_seed
from .losses import LossWeights, total_loss as training_total_loss


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
    accumulation_steps: int = 1
    use_scheduler: bool = True
    weights: LossWeights = field(default_factory=LossWeights)


class Trainer:
    """Research-grade trainer with deterministic seeds, checkpointing and seed-aware logging."""

    def __init__(self, model: torch.nn.Module, cfg, train_cfg: TrainerConfig):
        set_seed(train_cfg.seed)
        self.model = model
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(train_cfg.steps, 1)) if train_cfg.use_scheduler else None
        self.scaler = GradScaler(enabled=train_cfg.amp and self.device.type == "cuda")
        self.curriculum = Curriculum()
        self.history: list[dict] = []
        self.step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.stopped_early = False
        self._accum = 0
        self._running_loss = 0.0
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(train_cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def sample_task(self) -> str:
        return self.train_cfg.task if self.train_cfg.task != "mixed" else self.curriculum.sample()

    def batch(self):
        task = self.sample_task()
        return task, sample_batch(task, batch=self.train_cfg.batch_size, vocab_size=self.cfg.vocab_size)

    def _forward(self, batch):
        tokens = batch.tokens.to(self.device)
        numeric_x = batch.numeric_x.to(self.device)
        labels = batch.labels.to(self.device)
        rubric = batch.rubric.to(self.device)
        outputs = self.model(tokens, numeric_x=numeric_x, steps=0)
        loss, stats = training_total_loss(outputs, labels, rubric, weights=self.train_cfg.weights)
        return loss, outputs, stats, labels

    def train_step(self) -> Dict[str, float]:
        self.model.train()
        task, batch = self.batch()
        with autocast(enabled=self.scaler.is_enabled()):
            loss, outputs, stats, labels = self._forward(batch)
            loss = loss / max(self.train_cfg.accumulation_steps, 1)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum += 1
        self._running_loss += float(loss.detach().cpu())
        stepped = False
        if self._accum >= max(self.train_cfg.accumulation_steps, 1):
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            if hasattr(self.model, "update_target"):
                try:
                    self.model.update_target()
                except Exception:
                    pass
            self._accum = 0
            stepped = True

        pred = outputs["logits"].argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        self.curriculum.update(acc)
        record = {"step": self.step, "task": task, "acc": acc, "loss": float(stats["total"]), **stats}
        if stepped:
            record["optimizer_step"] = True
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
            confs.append(out.get("confidence", torch.zeros_like(out["logits"][:, :1])).detach().cpu())
            rubrics.append(out.get("rubric", batch.rubric).detach().cpu())
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        conf = torch.cat(confs).mean().item()
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc, "confidence": conf}

    def fit(self) -> torch.nn.Module:
        for _ in range(self.train_cfg.steps):
            rec = self.train_step()
            if self.train_cfg.eval_every and (self.step % self.train_cfg.eval_every == 0):
                eval_score = self.evaluate("parity", batches=2)
                rec.update({"eval_accuracy": eval_score["accuracy"], "eval_confidence": eval_score["confidence"]})
                self.best_metric = max(self.best_metric, eval_score["accuracy"])
            if self.train_cfg.save_every and (self.step % self.train_cfg.save_every == 0):
                self.save("latest.pt")
            if self.train_cfg.early_stop_patience > 0 and len(self.history) > self.train_cfg.early_stop_patience:
                recent = [h["acc"] for h in self.history[-self.train_cfg.early_stop_patience:]]
                if sum(recent) / len(recent) < 0.15:
                    self.stopped_early = True
                    break
        self.save("final.pt")
        self.model.eval()
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
            extra={"config": asdict(self.cfg), "train_config": asdict(self.train_cfg), "curriculum_level": self.curriculum.level, **(extra or {})},
        )

    def load(self, path: str | Path):
        ckpt = load_checkpoint(path, self.model, self.opt, self.scheduler, map_location=self.device)
        self.step = int(ckpt.get("step", self.step))
        self.history = list(ckpt.get("metrics_history", self.history)) if isinstance(ckpt.get("metrics_history"), list) else self.history
        extra = ckpt.get("extra", {}) or {}
        if isinstance(extra, dict) and "curriculum_level" in extra:
            self.curriculum.level = int(extra["curriculum_level"])
        return ckpt

    def export_history(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return path
