from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopper:
    patience: int = 10
    min_delta: float = 0.0

    best: float | None = None
    bad_steps: int = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience
