from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from .latent_world_model import LatentBelief, LatentTransition, LatentWorldModel


@dataclass
class CounterfactualOutcome:
    action_name: str
    horizon: str
    utility: float
    retention: float
    confusion: float
    mastery_gain: float
    trajectory: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class CounterfactualEngine(nn.Module):
    """Simulates alternative tutoring interventions and compares their long-horizon effects."""

    def __init__(self, world_model: LatentWorldModel | None = None):
        super().__init__()
        self.world_model = world_model or LatentWorldModel()

    def simulate(
        self,
        state: torch.Tensor | dict | LatentBelief,
        action_set: Sequence[int | torch.Tensor | Sequence[int]],
        context: torch.Tensor | None = None,
        horizon: str = "long",
        temperature: float = 1.0,
        rollout_steps: int = 3,
    ) -> list[CounterfactualOutcome]:
        outcomes: list[CounterfactualOutcome] = []
        for action in action_set:
            trajectory = self.world_model.rollout(state, [action] * max(1, rollout_steps), context=context, horizon=horizon, temperature=temperature)
            final = trajectory[-1].belief if trajectory else (state if isinstance(state, LatentBelief) else self.world_model._belief_from_hidden(self.world_model.encode_state(state)))
            utility = float(self.world_model.utility(final).mean().detach().cpu())
            outcomes.append(
                CounterfactualOutcome(
                    action_name=trajectory[-1].action_name if trajectory else f"action_{action}",
                    horizon=horizon,
                    utility=utility,
                    retention=float(final.retention.mean().detach().cpu()),
                    confusion=float(final.uncertainty.mean().detach().cpu() + final.fatigue.mean().detach().cpu()),
                    mastery_gain=float((final.retention.mean() - final.uncertainty.mean()).detach().cpu()),
                    trajectory=[{
                        "step": step.step,
                        "score": step.score,
                        "retention": float(step.belief.retention.mean().detach().cpu()),
                        "uncertainty": float(step.belief.uncertainty.mean().detach().cpu()),
                    } for step in trajectory],
                    metadata={"temperature": temperature, "rollout_steps": rollout_steps},
                )
            )
        return outcomes

    def best_action(self, state: torch.Tensor | dict | LatentBelief, action_set: Sequence[int | torch.Tensor | Sequence[int]], **kwargs) -> CounterfactualOutcome:
        outcomes = self.simulate(state, action_set, **kwargs)
        outcomes.sort(key=lambda o: (o.utility, o.mastery_gain, -o.confusion), reverse=True)
        return outcomes[0]

    def report(self, outcomes: Sequence[CounterfactualOutcome]) -> dict:
        if not outcomes:
            return {"count": 0}
        utilities = torch.tensor([o.utility for o in outcomes])
        gains = torch.tensor([o.mastery_gain for o in outcomes])
        confusions = torch.tensor([o.confusion for o in outcomes])
        return {
            "count": len(outcomes),
            "best_action": max(outcomes, key=lambda o: o.utility).action_name,
            "mean_utility": float(utilities.mean().item()),
            "mean_mastery_gain": float(gains.mean().item()),
            "mean_confusion": float(confusions.mean().item()),
        }
