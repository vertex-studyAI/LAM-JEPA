from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_mlp, clip_prob, ensure_2d, normalize, safe_mean


@dataclass
class LatentBelief:
    mean: torch.Tensor
    logvar: torch.Tensor
    uncertainty: torch.Tensor
    fatigue: torch.Tensor
    retention: torch.Tensor
    velocity: torch.Tensor
    hidden: torch.Tensor
    metadata: dict = field(default_factory=dict)

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)


@dataclass
class LatentTransition:
    current: LatentBelief
    next: LatentBelief
    action_name: str
    action_vector: torch.Tensor
    utility: torch.Tensor
    rollout_score: torch.Tensor


@dataclass
class LatentImaginationStep:
    step: int
    belief: LatentBelief
    action_name: str
    score: float


class LatentWorldModel(nn.Module):
    """Hierarchical latent transition model for student learning trajectories."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        action_dim: int = 32,
        context_dim: int = 32,
        num_actions: int = 12,
        short_horizon: int = 1,
        long_horizon: int = 6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon

        self.state_encoder = build_mlp(latent_dim, hidden_dim, hidden_dim, depth=3)
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.context_proj = build_mlp(context_dim, hidden_dim, hidden_dim, depth=2)

        self.short_transition = build_mlp(hidden_dim + action_dim + hidden_dim, hidden_dim, hidden_dim, depth=3)
        self.long_transition = build_mlp(hidden_dim + action_dim + hidden_dim, hidden_dim, hidden_dim, depth=3)
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.uncertainty_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.fatigue_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.retention_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.velocity_head = build_mlp(hidden_dim, hidden_dim, latent_dim, depth=2)
        self.utility_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.hidden_head = build_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

    def _context(self, context: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if context is None:
            return torch.zeros(batch, self.hidden_dim, device=device)
        context = ensure_2d(context.to(device))
        if context.size(-1) < self.context_dim:
            context = F.pad(context, (0, self.context_dim - context.size(-1)))
        context = context[..., : self.context_dim]
        return self.context_proj(context)

    def _state_tensor(self, state: torch.Tensor | dict | LatentBelief) -> tuple[torch.Tensor, dict]:
        if isinstance(state, LatentBelief):
            return state.mean, state.metadata
        if isinstance(state, dict):
            for key in ("latent_state", "mean", "state", "hidden"):
                if key in state and isinstance(state[key], torch.Tensor):
                    return state[key], dict(state.get("metadata", {}))
            raise KeyError("state dict must contain a latent tensor")
        return state, {}

    def encode_state(self, state: torch.Tensor | dict | LatentBelief) -> torch.Tensor:
        tensor, _ = self._state_tensor(state)
        tensor = ensure_2d(tensor)
        if tensor.size(-1) < self.latent_dim:
            tensor = F.pad(tensor, (0, self.latent_dim - tensor.size(-1)))
        tensor = tensor[..., : self.latent_dim]
        return self.state_encoder(tensor)

    def _belief_from_hidden(self, hidden: torch.Tensor, metadata: dict | None = None) -> LatentBelief:
        mean = self.mean_head(hidden)
        logvar = torch.tanh(self.logvar_head(hidden)) * 1.5 - 1.5
        uncertainty = clip_prob(torch.sigmoid(self.uncertainty_head(hidden)))
        fatigue = clip_prob(torch.sigmoid(self.fatigue_head(hidden)))
        retention = clip_prob(torch.sigmoid(self.retention_head(hidden)))
        velocity = self.velocity_head(hidden)
        return LatentBelief(
            mean=mean,
            logvar=logvar,
            uncertainty=uncertainty,
            fatigue=fatigue,
            retention=retention,
            velocity=velocity,
            hidden=self.hidden_head(hidden),
            metadata=metadata or {},
        )

    def transition(
        self,
        state: torch.Tensor | dict | LatentBelief,
        action: torch.Tensor | int | Sequence[int] | None = None,
        context: torch.Tensor | None = None,
        horizon: str = "short",
    ) -> LatentBelief:
        state_tensor, metadata = self._state_tensor(state)
        state_enc = self.encode_state(state_tensor)
        batch = state_enc.size(0)
        device = state_enc.device

        if action is None:
            action_vec = torch.zeros(batch, self.action_dim, device=device)
            action_name = "null"
        elif isinstance(action, int):
            action_idx = torch.full((batch,), int(action), device=device, dtype=torch.long)
            action_vec = self.action_embed(action_idx)
            action_name = f"action_{int(action)}"
        elif isinstance(action, Sequence) and not isinstance(action, torch.Tensor):
            action_idx = torch.tensor(list(action), device=device, dtype=torch.long)
            action_vec = self.action_embed(action_idx)
            action_name = "batched_action"
        else:
            action_tensor = ensure_2d(torch.as_tensor(action, device=device))
            if action_tensor.size(-1) < self.action_dim:
                action_tensor = F.pad(action_tensor, (0, self.action_dim - action_tensor.size(-1)))
            action_vec = action_tensor[..., : self.action_dim]
            action_name = "provided_action"

        ctx = self._context(context, batch=batch, device=device)
        h = torch.cat([state_enc, action_vec, ctx], dim=-1)
        if horizon == "long":
            hidden = self.long_transition(h)
        else:
            hidden = self.short_transition(h)
        return self._belief_from_hidden(hidden, metadata={**metadata, "action_name": action_name, "horizon": horizon})

    def sample_transition(
        self,
        state: torch.Tensor | dict | LatentBelief,
        action: torch.Tensor | int | Sequence[int] | None = None,
        context: torch.Tensor | None = None,
        horizon: str = "short",
        temperature: float = 1.0,
    ) -> LatentTransition:
        current = state if isinstance(state, LatentBelief) else self._belief_from_hidden(self.encode_state(state))
        nxt = self.transition(current, action=action, context=context, horizon=horizon)
        std = (0.5 * nxt.logvar).exp() * max(float(temperature), 1e-6)
        noise = torch.randn_like(nxt.mean) * std
        sampled_mean = nxt.mean + noise
        sampled = LatentBelief(
            mean=sampled_mean,
            logvar=nxt.logvar,
            uncertainty=nxt.uncertainty,
            fatigue=nxt.fatigue,
            retention=nxt.retention,
            velocity=nxt.velocity,
            hidden=nxt.hidden,
            metadata=dict(nxt.metadata),
        )
        utility = torch.sigmoid(self.utility_head(nxt.hidden))
        rollout_score = utility + nxt.retention - nxt.uncertainty - 0.5 * nxt.fatigue
        return LatentTransition(current=current, next=sampled, action_name=nxt.metadata.get("action_name", "action"), action_vector=torch.zeros_like(nxt.mean[..., : self.action_dim]), utility=utility, rollout_score=rollout_score)

    def rollout(
        self,
        state: torch.Tensor | dict | LatentBelief,
        actions: Iterable[torch.Tensor | int | Sequence[int]],
        context: torch.Tensor | None = None,
        horizon: str = "short",
        temperature: float = 1.0,
    ) -> list[LatentImaginationStep]:
        belief = state if isinstance(state, LatentBelief) else self._belief_from_hidden(self.encode_state(state))
        trajectory: list[LatentImaginationStep] = []
        running = belief
        for step, action in enumerate(actions):
            trans = self.sample_transition(running, action=action, context=context, horizon=horizon, temperature=temperature)
            running = trans.next
            trajectory.append(LatentImaginationStep(step=step, belief=running, action_name=trans.action_name, score=float(trans.rollout_score.mean().detach().cpu())))
        return trajectory

    def imagine_actions(
        self,
        state: torch.Tensor | dict | LatentBelief,
        candidate_actions: Sequence[torch.Tensor | int | Sequence[int]],
        context: torch.Tensor | None = None,
        horizon: str = "long",
        temperature: float = 1.0,
    ) -> list[LatentTransition]:
        transitions: list[LatentTransition] = []
        for action in candidate_actions:
            transitions.append(self.sample_transition(state, action=action, context=context, horizon=horizon, temperature=temperature))
        return transitions

    def forward(self, state: torch.Tensor | dict | LatentBelief, action: torch.Tensor | int | Sequence[int] | None = None, context: torch.Tensor | None = None, horizon: str = "short") -> LatentBelief:
        return self.transition(state, action=action, context=context, horizon=horizon)

    def utility(self, belief: LatentBelief) -> torch.Tensor:
        return torch.sigmoid(self.utility_head(belief.hidden)) + belief.retention - belief.uncertainty - 0.5 * belief.fatigue

    def summary(self, belief: LatentBelief) -> dict:
        return {
            "mean_norm": float(belief.mean.norm(dim=-1).mean().detach().cpu()),
            "uncertainty": float(safe_mean(belief.uncertainty).detach().cpu()),
            "fatigue": float(safe_mean(belief.fatigue).detach().cpu()),
            "retention": float(safe_mean(belief.retention).detach().cpu()),
            "velocity": float(belief.velocity.norm(dim=-1).mean().detach().cpu()),
        }
