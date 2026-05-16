from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass
class GrokkingSummary:
    train_acc: float
    test_acc: float
    gap: float
    phase: str


def summarize_grokking(train_acc: float, test_acc: float) -> GrokkingSummary:
    gap = float(train_acc - test_acc)
    if gap < 0.05 and test_acc > 0.8:
        phase = "generalized"
    elif gap > 0.25 and train_acc > 0.9:
        phase = "memorizing"
    else:
        phase = "transitioning"
    return GrokkingSummary(train_acc=train_acc, test_acc=test_acc, gap=gap, phase=phase)
