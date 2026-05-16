from __future__ import annotations
from dataclasses import dataclass

from .trainer import Trainer, TrainerConfig


@dataclass
class MultiTaskTrainerConfig(TrainerConfig):
    tasks: tuple[str, ...] = ("parity", "modadd", "chain", "algebra")


class MultiTaskTrainer(Trainer):
    def sample_task(self) -> str:
        idx = self.step % len(getattr(self.train_cfg, "tasks", ("parity",)))
        return getattr(self.train_cfg, "tasks", ("parity",))[idx]
