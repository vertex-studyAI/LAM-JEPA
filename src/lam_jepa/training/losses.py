from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    ce: float = 1.0
    alignment: float = 0.5
    variance: float = 0.1
    covariance: float = 0.05
    uniformity: float = 0.05
    trajectory: float = 0.2
    calibration: float = 0.3
    verifier: float = 0.3
    rubric: float = 0.2
    uncertainty: float = 0.1
    planning: float = 0.1


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.ndim < 2 or z.size(0) < 2:
        return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.ndim < 2 or z.size(0) < 2:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / max(z.shape[1], 1)


def uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    return torch.pdist(z, p=2).pow(2).mul(-2).exp().mean().log()


def trajectory_smoothness(traj: list[torch.Tensor]) -> torch.Tensor:
    if len(traj) < 2:
        return traj[0].new_tensor(0.0) if traj else torch.tensor(0.0)
    penalties = []
    for a, b in zip(traj[:-1], traj[1:]):
        penalties.append(F.mse_loss(a, b.detach()))
    return torch.stack(penalties).mean()


def calibration_loss(confidence: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(confidence.squeeze(-1), correct.float())


def verifier_loss(verifier: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(verifier.squeeze(-1).clamp(1e-4, 1 - 1e-4), correct.float())


def rubric_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def uncertainty_regularization(z: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    if z.ndim < 2:
        return z.new_tensor(0.0)
    return (z.norm(dim=-1).mean() * (1.0 - confidence.mean())).abs()


def planning_coherence(z: torch.Tensor, trajectory: list[torch.Tensor]) -> torch.Tensor:
    if not trajectory:
        return z.new_tensor(0.0)
    target = z.detach()
    return sum(F.mse_loss(step, target) for step in trajectory) / len(trajectory)


def total_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, rubric_target: Optional[torch.Tensor] = None, weights: LossWeights | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = weights or LossWeights()
    logits = outputs["logits"]
    z = outputs["z"]
    z_q = outputs.get("z_q", z)
    t_z = outputs.get("target_z", z.detach())
    traj = outputs.get("traj", [z])
    confidence = outputs.get("confidence", torch.sigmoid(logits.new_zeros(logits.size(0), 1)))
    verifier = outputs.get("verifier", confidence)
    rubric = outputs.get("rubric", logits.new_zeros(logits.size(0), 4))
    quant_loss = outputs.get("quant_loss", logits.new_tensor(0.0))

    ce = F.cross_entropy(logits, labels)
    align = cosine_alignment(z_q, t_z)
    var = variance_loss(z_q)
    cov = covariance_loss(z_q)
    uni = uniformity_loss(z_q)
    pred_correct = logits.argmax(dim=-1).eq(labels)
    conf = calibration_loss(confidence, pred_correct)
    ver = verifier_loss(verifier, pred_correct)

    traj_loss = trajectory_smoothness(traj)
    if rubric_target is None:
        rubric_target = torch.zeros_like(rubric)
    rub = rubric_loss(rubric, rubric_target)
    unc = uncertainty_regularization(z_q, confidence)
    plan = planning_coherence(z_q, traj)

    total = (
        weights.ce * ce
        + weights.alignment * align
        + weights.variance * var
        + weights.covariance * cov
        + weights.uniformity * uni
        + weights.trajectory * traj_loss
        + weights.calibration * conf
        + weights.verifier * ver
        + weights.rubric * rub
        + weights.uncertainty * unc
        + weights.planning * plan
        + 0.1 * quant_loss
    )
    stats = {
        "ce": float(ce.detach().cpu()),
        "align": float(align.detach().cpu()),
        "var": float(var.detach().cpu()),
        "cov": float(cov.detach().cpu()),
        "uni": float(uni.detach().cpu()),
        "traj": float(traj_loss.detach().cpu()),
        "conf": float(conf.detach().cpu()),
        "ver": float(ver.detach().cpu()),
        "rub": float(rub.detach().cpu()),
        "unc": float(unc.detach().cpu()),
        "plan": float(plan.detach().cpu()),
        "quant": float(quant_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, stats
