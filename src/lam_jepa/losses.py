from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def cosine_alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / z.shape[1]


def uniformity_loss(z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    return torch.pdist(z, p=2).pow(2).mul(-2).exp().mean().log()


def geodesic_penalty(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    return torch.cdist(z, z, p=2).mean()


def calibration_loss(confidence: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(confidence.squeeze(-1), correct.float())


def verifier_loss(verifier: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(verifier.squeeze(-1), correct.float())


def rubric_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def total_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, rubric_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    z = outputs["z"]
    z_q = outputs["z_q"]
    t_z = outputs["target_z"]
    traj = outputs["traj"]
    confidence = outputs["confidence"]
    verifier = outputs["verifier"]
    rubric = outputs["rubric"]
    quant_loss = outputs["quant_loss"]

    ce = F.cross_entropy(logits, labels)
    align = cosine_alignment(z_q, t_z)
    var = variance_loss(z_q)
    cov = covariance_loss(z_q)
    uni = uniformity_loss(z_q)
    geo = geodesic_penalty(z_q)
    pred_correct = logits.argmax(dim=-1).eq(labels)
    conf = calibration_loss(confidence, pred_correct)
    ver = verifier_loss(verifier, pred_correct)

    traj_loss = z.new_tensor(0.0)
    if len(traj) > 1:
        for z_h in traj[1:]:
            traj_loss = traj_loss + F.mse_loss(z_h, z_q.detach())
        traj_loss = traj_loss / max(len(traj) - 1, 1)

    if rubric_target is None:
        rubric_target = torch.zeros_like(rubric)
    rub = rubric_loss(rubric, rubric_target)

    total = (
        ce
        + 0.5 * align
        + 0.1 * var
        + 0.05 * cov
        + 0.05 * uni
        + 0.05 * geo
        + 0.5 * conf
        + 0.5 * ver
        + 0.25 * traj_loss
        + 0.25 * rub
        + 0.25 * quant_loss
    )
    stats = {
        "ce": float(ce.detach().cpu()),
        "align": float(align.detach().cpu()),
        "var": float(var.detach().cpu()),
        "cov": float(cov.detach().cpu()),
        "uni": float(uni.detach().cpu()),
        "geo": float(geo.detach().cpu()),
        "conf": float(conf.detach().cpu()),
        "ver": float(ver.detach().cpu()),
        "traj": float(traj_loss.detach().cpu()),
        "rub": float(rub.detach().cpu()),
        "quant": float(quant_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, stats
