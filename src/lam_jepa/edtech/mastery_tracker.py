from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import torch


@dataclass
class ConceptStats:
    alpha: float = 1.0
    beta: float = 1.0
    forgetting: float = 0.02
    last_correct: float = 0.0
    last_difficulty: float = 0.0
    last_timestamp: float = 0.0
    confidence: float = 0.5
    retention: float = 0.5

    @property
    def mastery(self) -> float:
        return self.alpha / max(self.alpha + self.beta, 1e-6)

    @property
    def uncertainty(self) -> float:
        total = self.alpha + self.beta
        return float((self.alpha * self.beta) / max(total * total * (total + 1.0), 1e-6))


class MasteryTracker:
    """Bayesian mastery tracker with forgetting curves, confidence and retention estimates."""

    def __init__(self, num_concepts: int = 64, prior: float = 1.0, forgetting: float = 0.02):
        self.num_concepts = num_concepts
        self.stats = [ConceptStats(alpha=prior, beta=prior, forgetting=forgetting) for _ in range(num_concepts)]
        self.velocity = torch.zeros(num_concepts)
        self.stability = torch.ones(num_concepts) * 0.5
        self._time = 0.0

    def apply_forgetting(self, elapsed: float = 1.0) -> None:
        decay = math.exp(-float(elapsed) * 0.05)
        for stat in self.stats:
            stat.alpha = 1.0 + (stat.alpha - 1.0) * decay
            stat.beta = 1.0 + (stat.beta - 1.0) * decay
            stat.retention = float(stat.retention * decay)
            stat.confidence = float(stat.confidence * decay + 0.5 * (1.0 - decay))

    def update(self, concept_id: int, correct: float, difficulty: float = 0.5, confidence: float = 0.5, timestamp: float | None = None) -> None:
        concept_id = int(max(0, min(self.num_concepts - 1, concept_id)))
        stat = self.stats[concept_id]
        correct = float(correct)
        difficulty = float(difficulty)
        confidence = float(confidence)
        timestamp = float(self._time if timestamp is None else timestamp)
        elapsed = max(0.0, timestamp - stat.last_timestamp)
        if elapsed > 0:
            decay = math.exp(-elapsed * stat.forgetting)
            stat.alpha = 1.0 + (stat.alpha - 1.0) * decay
            stat.beta = 1.0 + (stat.beta - 1.0) * decay
        old_mastery = stat.mastery
        stat.alpha += 0.5 + 1.0 * correct + 0.25 * confidence + 0.1 * (1.0 - difficulty)
        stat.beta += 0.5 + 1.0 * (1.0 - correct) + 0.3 * difficulty * (1.0 - confidence)
        stat.last_correct = correct
        stat.last_difficulty = difficulty
        stat.last_timestamp = timestamp
        stat.confidence = float(0.7 * stat.confidence + 0.3 * confidence)
        stat.retention = float(0.75 * stat.retention + 0.25 * (0.6 + 0.4 * correct))
        new_mastery = stat.mastery
        self.velocity[concept_id] = 0.85 * self.velocity[concept_id] + 0.15 * (new_mastery - old_mastery)
        self.stability[concept_id] = 0.9 * self.stability[concept_id] + 0.1 * (1.0 - abs(new_mastery - old_mastery))
        self._time = max(self._time, timestamp + 1.0)

    def bulk_update(self, observations: Iterable[tuple[int, float, float, float]]) -> None:
        for concept_id, correct, difficulty, confidence in observations:
            self.update(concept_id, correct, difficulty=difficulty, confidence=confidence)

    def mastery_vector(self) -> torch.Tensor:
        return torch.tensor([s.mastery for s in self.stats], dtype=torch.float32)

    def uncertainty_vector(self) -> torch.Tensor:
        return torch.tensor([s.uncertainty for s in self.stats], dtype=torch.float32)

    def retention_vector(self) -> torch.Tensor:
        return torch.tensor([s.retention for s in self.stats], dtype=torch.float32)

    def next_target(self) -> int:
        mastery = self.mastery_vector()
        score = (1.0 - mastery) + 0.35 * self.uncertainty_vector() + 0.25 * self.velocity.abs() + 0.15 * (1.0 - self.stability)
        return int(torch.argmax(score).item())

    def summary(self) -> dict:
        mastery = self.mastery_vector()
        return {
            "mean_mastery": float(mastery.mean().item()),
            "min_mastery": float(mastery.min().item()),
            "max_mastery": float(mastery.max().item()),
            "mean_uncertainty": float(self.uncertainty_vector().mean().item()),
            "mean_retention": float(self.retention_vector().mean().item()),
            "mean_stability": float(self.stability.mean().item()),
            "fastest_growth": int(torch.argmax(self.velocity).item()),
            "slowest_growth": int(torch.argmin(self.velocity).item()),
        }
