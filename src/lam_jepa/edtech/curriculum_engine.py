from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .mastery_tracker import MasteryTracker


@dataclass
class CurriculumPlan:
    concept_id: int
    difficulty: float
    rationale: str
    mode: str
    priority: float = 1.0
    prerequisites: tuple[int, ...] = ()
    next_review_in: int = 1
    metadata: dict = field(default_factory=dict)


class CurriculumEngine:
    """Adaptive curriculum optimizer over a concept graph."""

    def __init__(
        self,
        num_concepts: int = 64,
        min_difficulty: float = 0.1,
        max_difficulty: float = 0.98,
        novelty_bonus: float = 0.15,
        review_bonus: float = 0.1,
    ):
        self.tracker = MasteryTracker(num_concepts=num_concepts)
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.novelty_bonus = novelty_bonus
        self.review_bonus = review_bonus
        self.num_concepts = num_concepts
        self.prerequisites = torch.zeros(num_concepts, num_concepts)
        self.concept_tags = [f"concept_{i}" for i in range(num_concepts)]

    def set_prerequisites(self, matrix: torch.Tensor) -> None:
        if matrix.shape != (self.num_concepts, self.num_concepts):
            raise ValueError("prerequisite matrix must be square with size num_concepts")
        self.prerequisites = matrix.float().clone()

    def update_from_result(self, concept_id: int, correct: float, difficulty: float, confidence: float) -> None:
        self.tracker.update(concept_id, correct, difficulty=difficulty, confidence=confidence)

    def _prereq_score(self, concept_id: int) -> float:
        prereqs = self.prerequisites[concept_id]
        if prereqs.sum() <= 0:
            return 1.0
        mastery = self.tracker.mastery_vector()
        satisfied = float((mastery * prereqs).sum().item() / max(prereqs.sum().item(), 1e-6))
        return satisfied

    def select_next(self, preferred_concepts: Sequence[int] | None = None) -> CurriculumPlan:
        mastery = self.tracker.mastery_vector()
        uncertainty = self.tracker.uncertainty_vector()
        velocity = self.tracker.velocity
        retention = self.tracker.retention_vector()

        score = (1.0 - mastery) + 0.35 * uncertainty + self.novelty_bonus * velocity.abs() + 0.2 * (1.0 - retention) + self.review_bonus * (1.0 - self.tracker.stability)
        if preferred_concepts:
            mask = torch.zeros_like(score)
            mask[list(preferred_concepts)] = 1.0
            score = score * (0.85 + 0.15 * mask)
        prereq = torch.tensor([self._prereq_score(i) for i in range(self.num_concepts)], dtype=torch.float32)
        score = score * (0.65 + 0.35 * prereq)
        concept_id = int(torch.argmax(score).item())
        m = float(mastery[concept_id].item())
        u = float(uncertainty[concept_id].item())
        raw_difficulty = 0.2 + 0.6 * m + 0.15 * u
        difficulty = float(max(self.min_difficulty, min(self.max_difficulty, raw_difficulty)))
        if m < 0.35:
            mode = "scaffold"
            rationale = "mastery is low, so a guided explanation should precede independent practice"
        elif m < 0.7:
            mode = "practice"
            rationale = "the concept is emerging and should be strengthened with targeted retrieval"
        else:
            mode = "challenge"
            rationale = "the learner is likely ready for transfer and harder variants"
        prereq_ids = tuple(int(i) for i in torch.nonzero(self.prerequisites[concept_id] > 0, as_tuple=False).flatten().tolist())
        return CurriculumPlan(concept_id=concept_id, difficulty=difficulty, rationale=rationale, mode=mode, priority=float(score[concept_id].item()), prerequisites=prereq_ids, next_review_in=max(1, int(round((1.0 - m) * 5))), metadata={"mastery": m, "uncertainty": u})

    def batch_plan(self, k: int = 4) -> list[CurriculumPlan]:
        plans = []
        snapshot = self.tracker.mastery_vector().clone()
        for _ in range(k):
            plan = self.select_next()
            plans.append(plan)
            self.tracker.update(plan.concept_id, correct=1.0, difficulty=plan.difficulty, confidence=0.75)
        for idx, mastery in enumerate(snapshot):
            self.tracker.stats[idx].alpha = float(mastery.item()) + 1.0
            self.tracker.stats[idx].beta = max(1.0, 2.0 - float(mastery.item()))
        return plans

    def plan_path(self, length: int = 5) -> list[CurriculumPlan]:
        path = []
        for _ in range(length):
            plan = self.select_next()
            path.append(plan)
            self.update_from_result(plan.concept_id, correct=1.0, difficulty=plan.difficulty, confidence=0.7)
        return path
