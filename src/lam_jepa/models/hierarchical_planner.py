from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_mlp
from .latent_world_model import LatentBelief, LatentWorldModel
from .student_state_space import StudentManifoldPoint, StudentStateSpace


@dataclass
class PlannerAction:
    name: str
    concept_id: int
    difficulty: float
    utility: float
    confidence: float
    rollout_score: float
    rationale: str
    horizon: str = "short"
    metadata: dict = field(default_factory=dict)


@dataclass
class PlannerPlan:
    actions: list[PlannerAction]
    predicted_trajectory: list[dict]
    best_action: PlannerAction
    utility: float
    uncertainty: float


class HierarchicalPlanner(nn.Module):
    """Multi-horizon tutoring planner that searches over latent interventions."""

    def __init__(
        self,
        latent_dim: int = 128,
        num_concepts: int = 64,
        num_actions: int = 12,
    ):
        super().__init__()
        self.world_model = LatentWorldModel(latent_dim=latent_dim, num_actions=num_actions)
        self.state_space = StudentStateSpace(num_concepts=num_concepts, latent_dim=latent_dim)
        self.action_embed = nn.Embedding(num_actions, latent_dim)
        self.scorer = build_mlp(latent_dim * 2 + 4, latent_dim, 1, depth=3)
        self.num_concepts = num_concepts
        self.num_actions = num_actions

    def _action_name(self, action_id: int, concept_id: int | None = None) -> str:
        if concept_id is not None:
            return f"concept_{concept_id}_action_{action_id}"
        return f"action_{action_id}"

    def candidate_actions(self, concept_ids: Sequence[int] | None = None) -> list[tuple[int, int | None]]:
        if not concept_ids:
            return [(i, None) for i in range(min(self.num_actions, 8))]
        actions = []
        for c in concept_ids:
            actions.extend([(i, int(c)) for i in range(min(self.num_actions, 4))])
        return actions

    def _belief_from_point(self, point: StudentManifoldPoint) -> LatentBelief:
        hidden = point.latent
        return LatentBelief(
            mean=point.latent,
            logvar=torch.zeros_like(point.latent) - 1.0,
            uncertainty=point.uncertainty,
            fatigue=torch.zeros(point.latent.size(0), 1, device=point.latent.device),
            retention=point.confidence,
            velocity=torch.zeros_like(point.latent),
            hidden=hidden,
            metadata={"manifold": True},
        )

    def _score_transition(self, belief: LatentBelief, transition, manifold: StudentManifoldPoint | None = None) -> float:
        gain = transition.next.retention.mean() - transition.next.uncertainty.mean() - 0.5 * transition.next.fatigue.mean()
        if manifold is not None:
            gain = gain + 0.2 * manifold.confidence.mean() - 0.1 * manifold.curvature.mean()
        return float(gain.detach().cpu())

    def plan(
        self,
        student_latent: torch.Tensor | StudentManifoldPoint | LatentBelief,
        concept_ids: Sequence[int] | None = None,
        context: torch.Tensor | None = None,
        horizon: str = "long",
        temperature: float = 1.0,
    ) -> PlannerPlan:
        if isinstance(student_latent, StudentManifoldPoint):
            belief = self._belief_from_point(student_latent)
            manifold = student_latent
        elif isinstance(student_latent, LatentBelief):
            belief = student_latent
            manifold = None
        else:
            latent = student_latent if student_latent.ndim > 1 else student_latent.unsqueeze(0)
            belief = self.world_model._belief_from_hidden(self.world_model.encode_state(latent))
            manifold = None

        actions = self.candidate_actions(concept_ids)
        scored: list[PlannerAction] = []
        predicted: list[dict] = []
        for action_id, concept_id in actions:
            action_vec = self.action_embed(torch.tensor([action_id], device=belief.mean.device))
            transition = self.world_model.sample_transition(belief, action=action_id, context=context, horizon=horizon, temperature=temperature)
            state_point = self.state_space.decode(transition.next.mean)
            score = self._score_transition(belief, transition, manifold=manifold)
            utility = float((transition.utility.mean() + transition.next.retention.mean() - transition.next.uncertainty.mean()).detach().cpu())
            rationale = "increase mastery while controlling uncertainty"
            if state_point["curvature"].mean().item() > 0.4:
                rationale = "avoid overly sharp trajectory; prefer smoother support"
            action = PlannerAction(
                name=self._action_name(action_id, concept_id),
                concept_id=int(concept_id if concept_id is not None else action_id % self.num_concepts),
                difficulty=float(torch.sigmoid(action_vec.mean()).item()),
                utility=utility,
                confidence=float(transition.next.retention.mean().detach().cpu()),
                rollout_score=score,
                rationale=rationale,
                horizon=horizon,
                metadata={"state_point": {k: float(v.mean().detach().cpu()) for k, v in state_point.items()}},
            )
            scored.append(action)
            predicted.append({"action": action.name, "utility": action.utility, "confidence": action.confidence, "rollout_score": action.rollout_score})

        scored.sort(key=lambda a: (a.rollout_score, a.utility, a.confidence), reverse=True)
        best = scored[0]
        return PlannerPlan(actions=scored, predicted_trajectory=predicted, best_action=best, utility=best.utility, uncertainty=1.0 - best.confidence)

    def beam_search(self, student_latent: torch.Tensor | StudentManifoldPoint | LatentBelief, beam_width: int = 4, depth: int = 3) -> list[PlannerPlan]:
        plans = []
        current = student_latent
        for _ in range(depth):
            plan = self.plan(current)
            plans.append(plan)
            current = self.world_model.transition(current if isinstance(current, (LatentBelief, StudentManifoldPoint)) else current, action=int(plan.best_action.concept_id), horizon="long")
        return plans[:beam_width]
