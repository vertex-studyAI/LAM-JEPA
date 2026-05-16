from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .curriculum_engine import CurriculumEngine
from .intervention_selector import Intervention, InterventionSelector
from .misconception_model import MisconceptionModel
from .student_model import StudentInteraction, StudentModel, StudentState


@dataclass
class TutorDecision:
    concept_id: int
    difficulty: float
    intervention: Intervention
    state_summary: dict
    misconception_ranking: list[dict]
    counterfactuals: list[dict] = field(default_factory=list)
    utility: float = 0.0
    next_review_in: int = 1


class TutorPolicy:
    """Unified tutoring policy for adaptive instruction research."""

    def __init__(self, student_model: StudentModel | None = None, num_concepts: int = 64):
        self.student_model = student_model or StudentModel(num_concepts=num_concepts)
        self.curriculum = CurriculumEngine(num_concepts=num_concepts)
        self.misconception_model = MisconceptionModel()
        self.selector = InterventionSelector()
        self._last_state: StudentState | None = None

    @torch.no_grad()
    def observe(self, interactions: list[StudentInteraction]) -> StudentState:
        state = self.student_model.update_state(interactions)
        self._last_state = state
        for idx, interaction in enumerate(interactions):
            self.curriculum.update_from_result(
                interaction.concept_id,
                interaction.correct,
                difficulty=interaction.difficulty,
                confidence=interaction.confidence,
            )
            if idx > 0 and interactions[idx - 1].misconception_id >= 0 and interaction.misconception_id >= 0:
                prev = self.misconception_model.misconceptions[min(interactions[idx - 1].misconception_id, len(self.misconception_model.misconceptions) - 1)].name
                curr = self.misconception_model.misconceptions[min(interaction.misconception_id, len(self.misconception_model.misconceptions) - 1)].name
                self.misconception_model.update_transition(prev, curr)
        return state

    def _state_for_selection(self, state: StudentState) -> dict:
        return {
            "mastery": float(state.mastery.mean().item()),
            "confidence": float(state.confidence.mean().item()),
            "learning_velocity": float(state.learning_velocity.mean().item()),
            "uncertainty": float(state.uncertainty.mean().item()),
            "fatigue": float(state.fatigue.mean().item()),
            "retention": float(state.retention.mean().item()),
        }

    @torch.no_grad()
    def next_action(self, interactions: list[StudentInteraction] | None = None, task: str = "math") -> TutorDecision:
        if interactions:
            state = self.observe(interactions)
        else:
            state = self._last_state or self.student_model.update_state([])
        plan = self.curriculum.select_next()
        prompt_text = " ".join([i.metadata.get("prompt", "") for i in (interactions or [])])
        answer_text = " ".join([i.metadata.get("answer", "") for i in (interactions or [])])
        misconceptions = self.misconception_model.diagnose(task, prompt=prompt_text, answer=answer_text, predicted="")
        top_mc = misconceptions[0]["misconception"] if misconceptions else None
        intervention = self.selector.select(
            top_mc,
            confidence=float(state.confidence.mean().item()),
            mastery=float(state.mastery.mean().item()),
            difficulty=plan.difficulty,
            fatigue=float(state.fatigue.mean().item()),
            retention=float(state.retention.mean().item()),
        )
        counterfactuals = []
        for m in misconceptions[:3]:
            counterfactuals.append({
                "misconception": m["misconception"],
                "recommended_intervention": self.misconception_model.recommend_intervention(m["misconception"]),
                "probability": m["probability"],
            })
        utility = float(intervention.estimated_gain + 0.5 * float(state.retention.mean().item()) - 0.25 * float(state.fatigue.mean().item()))
        return TutorDecision(
            concept_id=plan.concept_id,
            difficulty=plan.difficulty,
            intervention=intervention,
            state_summary=self._state_for_selection(state),
            misconception_ranking=misconceptions,
            counterfactuals=counterfactuals,
            utility=utility,
            next_review_in=plan.next_review_in,
        )

    @torch.no_grad()
    def recommend_session(self, interactions: list[StudentInteraction] | None = None, task: str = "math") -> dict:
        decision = self.next_action(interactions=interactions, task=task)
        return {
            "concept_id": decision.concept_id,
            "difficulty": decision.difficulty,
            "intervention": decision.intervention.__dict__,
            "state_summary": decision.state_summary,
            "misconceptions": decision.misconception_ranking,
            "counterfactuals": decision.counterfactuals,
            "utility": decision.utility,
            "next_review_in": decision.next_review_in,
        }
