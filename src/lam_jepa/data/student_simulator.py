from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math
import random
import torch

from . import Batch


@dataclass
class StudentTrace:
    concept_id: int
    correct: float
    confidence: float
    fatigue: float
    retention: float
    misconception: str
    response_time: float
    metadata: dict = field(default_factory=dict)


class StudentSimulator:
    """Simulates longitudinal student behavior with misconceptions, fatigue and retention."""

    def __init__(self, num_concepts: int = 64, latent_dim: int = 16, seed: int = 7):
        self.num_concepts = num_concepts
        self.latent_dim = latent_dim
        self.rng = random.Random(seed)
        self.knowledge = torch.full((num_concepts,), 0.35, dtype=torch.float32)
        self.confidence = torch.full((num_concepts,), 0.5, dtype=torch.float32)
        self.retention = torch.full((num_concepts,), 0.6, dtype=torch.float32)
        self.fatigue = 0.1
        self.misconception_state = torch.zeros(num_concepts, dtype=torch.float32)
        self.learning_velocity = torch.zeros(num_concepts, dtype=torch.float32)

    def step(self, concept_id: int, difficulty: float, intervention_strength: float = 0.5) -> StudentTrace:
        idx = int(max(0, min(self.num_concepts - 1, concept_id)))
        mastery = float(self.knowledge[idx].item())
        confidence = float(self.confidence[idx].item())
        retention = float(self.retention[idx].item())
        fatigue = float(self.fatigue)
        confusion = float(self.misconception_state[idx].item())

        base_prob = 0.55 * mastery + 0.2 * confidence + 0.15 * retention - 0.15 * fatigue - 0.1 * confusion - 0.05 * difficulty
        base_prob += 0.12 * intervention_strength
        correct = 1.0 if self.rng.random() < max(0.0, min(1.0, base_prob)) else 0.0
        target = 1.0 if correct > 0.5 else 0.0

        update = 0.08 * (target - mastery) + 0.04 * intervention_strength - 0.03 * difficulty
        if correct < 0.5:
            update -= 0.05 * confusion
        self.knowledge[idx] = torch.clamp(self.knowledge[idx] + update, 0.0, 1.0)
        self.confidence[idx] = torch.clamp(0.9 * self.confidence[idx] + 0.1 * (0.5 + 0.5 * correct), 0.0, 1.0)
        self.retention[idx] = torch.clamp(0.95 * self.retention[idx] + 0.05 * (0.4 + 0.6 * correct), 0.0, 1.0)
        self.fatigue = float(max(0.0, min(1.0, 0.92 * self.fatigue + 0.08 * (0.25 + 0.4 * difficulty - 0.15 * intervention_strength))))
        self.misconception_state[idx] = torch.clamp(0.92 * self.misconception_state[idx] + 0.08 * (1.0 - correct) * (0.5 + difficulty), 0.0, 1.0)
        self.learning_velocity[idx] = 0.85 * self.learning_velocity[idx] + 0.15 * update

        response_time = max(0.5, 1.2 + 1.5 * difficulty - 0.8 * confidence + 0.4 * fatigue + self.rng.gauss(0.0, 0.1))
        misconception = "sign_error" if idx % 5 == 0 and correct < 0.5 else ("retrieval_gap" if correct < 0.5 else "none")
        return StudentTrace(
            concept_id=idx,
            correct=float(correct),
            confidence=float(self.confidence[idx].item()),
            fatigue=float(self.fatigue),
            retention=float(self.retention[idx].item()),
            misconception=misconception,
            response_time=float(response_time),
            metadata={"mastery": float(self.knowledge[idx].item()), "velocity": float(self.learning_velocity[idx].item())},
        )

    def simulate_session(self, batch: Batch | Sequence[Batch], intervention_strength: float = 0.5) -> list[StudentTrace]:
        items = [batch] if isinstance(batch, Batch) else list(batch)
        traces = []
        for item in items:
            concept_id = int(item.numeric_x[0, 0].item()) if item.numeric_x.ndim >= 2 else 0
            difficulty = float(item.difficulty or 0.5)
            traces.append(self.step(concept_id, difficulty=difficulty, intervention_strength=intervention_strength))
        return traces

    def state_vector(self) -> torch.Tensor:
        return torch.cat([self.knowledge, self.confidence, self.retention, self.misconception_state, self.learning_velocity, torch.tensor([self.fatigue])])

    def reset_fatigue(self) -> None:
        self.fatigue = max(0.0, self.fatigue * 0.5)
