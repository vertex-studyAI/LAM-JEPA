"""Opt-in deterministic action-branching planner, separate from frozen ARC.

The legacy planner unpacks four values from the current six-value transition
API and expands only one successor, so beam_width cannot create a search.
This development implementation expands every discrete action and keeps an
independent beam and ancestry for each example in a batch.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from .planner import PlanResult


@contextmanager
def _evaluation_mode(*models):
    flags = {}
    for model in models:
        for module in model.modules():
            flags.setdefault(module, module.training)
    try:
        for model in models:
            model.eval()
        yield
    finally:
        # Direct assignment preserves intentionally mixed child training modes.
        for module, training in flags.items():
            module.training = training


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@torch.no_grad()
def beam_plan_v2(
    latent_action_model, value_head, verifier_head, z: torch.Tensor,
    steps: int = 6, beam_width: int = 4, temperature: float = 0.7,
    return_all: bool = False, *, max_candidates: int = 65536,
    max_state_elements: int = 8_000_000, max_steps: int = 128,
):
    """Search all action branches with bounded storage and deterministic ties.

    The retained score uses the legacy declared per-step objective:
    0.6*value + 0.4*verifier - 0.01*latent norm. All actions are considered;
    policy temperature does not change ranking under this objective. Ties keep
    parent order, then ascending action ID. Returned trajectories and actions
    follow the winning path for each row, without batch-average decisions.
    """
    for name, value, minimum in (("steps", steps, 0), ("beam_width", beam_width, 1),
                                 ("max_candidates", max_candidates, 1),
                                 ("max_state_elements", max_state_elements, 1),
                                 ("max_steps", max_steps, 1)):
        _integer(name, value, minimum)
    if steps > max_steps:
        raise ValueError("requested steps exceed the declared search-depth budget")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if (not isinstance(z, torch.Tensor) or z.ndim != 2 or not all(z.shape)
            or not z.is_floating_point() or not torch.isfinite(z).all()):
        raise ValueError("z must be a finite, nonempty floating-point [batch, latent] tensor")
    num_actions = latent_action_model.action_embed.num_embeddings
    _integer("num_actions", num_actions, 1)
    batch, dim = z.shape
    states = z.unsqueeze(1)
    scores = z.new_zeros(batch, 1)
    trajectories = states.unsqueeze(2)
    actions = torch.empty(batch, 1, 0, device=z.device, dtype=torch.long)

    with _evaluation_mode(latent_action_model, value_head, verifier_head):
        for depth in range(steps):
            width = states.shape[1]
            count = batch * width * num_actions
            keep = min(beam_width, width * num_actions)
            elements = 2 * count * dim + batch * keep * ((depth + 2) * dim + depth + 1)
            if count > max_candidates or elements > max_state_elements:
                raise ValueError("beam expansion exceeds the declared candidate/state budget")
            parents = states.unsqueeze(2).expand(batch, width, num_actions, dim).reshape(-1, dim)
            proposed = torch.arange(num_actions, device=z.device).view(1, 1, -1)
            proposed = proposed.expand(batch, width, num_actions).reshape(-1)
            transition = latent_action_model.step(
                parents, temp=temperature, sample=False, action_override=proposed, noise_std=0.0,
            )
            if not isinstance(transition, (tuple, list)) or len(transition) < 2:
                raise ValueError("transition must return next states and executed action IDs")
            next_states, executed = transition[:2]
            if (next_states.shape != parents.shape or next_states.device != z.device
                    or next_states.dtype != z.dtype or not torch.isfinite(next_states).all()
                    or executed.shape != proposed.shape or not torch.equal(executed, proposed)):
                raise ValueError("transition changed shape/dtype/device, ignored actions, or produced non-finite states")
            value = value_head(next_states)
            verified = verifier_head(next_states)
            if value.shape not in ((count,), (count, 1)) or verified.shape not in ((count,), (count, 1)):
                raise ValueError("value/verifier heads must return one scalar per candidate")
            reward = .6 * value.reshape(-1) + .4 * verified.reshape(-1) - .01 * next_states.norm(dim=-1)
            candidates = scores.unsqueeze(-1) + reward.reshape(batch, width, num_actions)
            candidates = candidates.reshape(batch, -1)
            if not torch.isfinite(candidates).all():
                raise ValueError("non-finite candidate score")
            chosen = torch.argsort(candidates, dim=1, descending=True, stable=True)[:, :keep]
            parent_index = chosen // num_actions
            next_flat = next_states.reshape(batch, -1, dim)
            states = next_flat.gather(1, chosen.unsqueeze(-1).expand(-1, -1, dim))
            scores = candidates.gather(1, chosen)
            prior_trajectory = trajectories.gather(1, parent_index[:, :, None, None].expand(
                -1, -1, trajectories.shape[2], dim))
            trajectories = torch.cat((prior_trajectory, states.unsqueeze(2)), dim=2)
            prior_actions = actions.gather(1, parent_index.unsqueeze(-1).expand(-1, -1, depth))
            actions = torch.cat((prior_actions, (chosen % num_actions).unsqueeze(-1)), dim=2)

    final = states[:, 0]
    if not return_all:
        return final
    return PlanResult(
        trajectory=list(trajectories[:, 0].unbind(dim=1)),
        actions=list(actions[:, 0].unbind(dim=1)),
        score=scores[:, 0], final_state=final,
    )
