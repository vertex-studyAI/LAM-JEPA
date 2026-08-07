from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..model import LAMJEPAConfig, MultiViewEncoder


@dataclass(frozen=True)
class MatchedCapacitySpec:
    target_parameters: int
    actual_parameters: int
    parameter_ratio: float
    hidden_width: int
    allowed_ratio_min: float
    allowed_ratio_max: float


class MatchedCapacityARCClassifier(nn.Module):
    """Non-JEPA supervised ARC baseline whose entire parameter set is on the answer-loss path.

    The encoder family is shared with the existing scratch baseline for input parity. Capacity is
    added with a residual supervised MLP. This class deliberately contains no target encoder,
    JEPA loss, latent-action planner, sparse memory, or vector quantizer.
    """

    def __init__(self, cfg: LAMJEPAConfig, *, hidden_width: int, num_choices: int = 4):
        super().__init__()
        if hidden_width < 1:
            raise ValueError("hidden_width must be >= 1")
        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.capacity_mlp = nn.Sequential(
            nn.Linear(cfg.proj_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
        )
        self.choice_head = nn.Linear(cfg.proj_dim, num_choices)

    def forward(self, tokens: torch.Tensor, numeric_x: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.encoder(tokens, numeric_x=numeric_x))
        z = z + self.capacity_mlp(z)
        return self.choice_head(z)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_matched_capacity_arc_classifier(
    cfg: LAMJEPAConfig,
    *,
    target_parameters: int,
    num_choices: int = 4,
    allowed_ratio_min: float = 0.99,
    allowed_ratio_max: float = 1.01,
    max_hidden_width: int = 4096,
) -> tuple[MatchedCapacityARCClassifier, MatchedCapacitySpec]:
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    if not (0.0 < allowed_ratio_min <= 1.0 <= allowed_ratio_max):
        raise ValueError("allowed ratio interval must contain 1.0")

    best_allowed: tuple[float, int, int] | None = None
    nearest: tuple[float, int, int] | None = None
    for hidden_width in range(1, max_hidden_width + 1):
        candidate = MatchedCapacityARCClassifier(
            cfg,
            hidden_width=hidden_width,
            num_choices=num_choices,
        )
        count = trainable_parameter_count(candidate)
        ratio = count / target_parameters
        distance = abs(1.0 - ratio)
        if nearest is None or distance < nearest[0]:
            nearest = (distance, hidden_width, count)
        if allowed_ratio_min <= ratio <= allowed_ratio_max:
            if best_allowed is None or distance < best_allowed[0]:
                best_allowed = (distance, hidden_width, count)
        elif ratio > allowed_ratio_max and best_allowed is not None:
            break

    if best_allowed is None:
        assert nearest is not None
        _, hidden_width, count = nearest
        raise RuntimeError(
            "could not construct a matched-capacity ARC baseline within the allowed ratio: "
            f"target={target_parameters}, nearest={count}, width={hidden_width}, "
            f"ratio={count / target_parameters:.6f}"
        )

    _, hidden_width, count = best_allowed
    model = MatchedCapacityARCClassifier(
        cfg,
        hidden_width=hidden_width,
        num_choices=num_choices,
    )
    spec = MatchedCapacitySpec(
        target_parameters=target_parameters,
        actual_parameters=count,
        parameter_ratio=count / target_parameters,
        hidden_width=hidden_width,
        allowed_ratio_min=allowed_ratio_min,
        allowed_ratio_max=allowed_ratio_max,
    )
    return model, spec


def gradient_active_parameter_count(model: nn.Module) -> tuple[int, list[str], list[str]]:
    active: list[str] = []
    inactive: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            inactive.append(name)
        else:
            active.append(name)
            count += int(parameter.numel())
    return count, active, inactive
