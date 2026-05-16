from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import build_mlp, clip_prob, normalize


@dataclass
class StudentManifoldPoint:
    latent: torch.Tensor
    mastery: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    curvature: torch.Tensor
    concept_logits: torch.Tensor
    metadata: dict


class StudentStateSpace(nn.Module):
    """Structured manifold for student learning states."""

    def __init__(self, num_concepts: int = 64, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.num_concepts = num_concepts
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.concept_embed = nn.Embedding(num_concepts, latent_dim)
        self.encoder = build_mlp(latent_dim + 5, hidden_dim, hidden_dim, depth=3)
        self.to_latent = build_mlp(hidden_dim, hidden_dim, latent_dim, depth=2)
        self.mastery_head = build_mlp(latent_dim, hidden_dim, num_concepts, depth=2)
        self.confidence_head = build_mlp(latent_dim, hidden_dim, 1, depth=2)
        self.uncertainty_head = build_mlp(latent_dim, hidden_dim, 1, depth=2)
        self.curvature_head = build_mlp(latent_dim, hidden_dim, 1, depth=2)
        self.decoder = build_mlp(latent_dim, hidden_dim, latent_dim, depth=2)

        prereq = torch.eye(num_concepts)
        if num_concepts > 1:
            prereq = prereq * 0.7 + torch.roll(prereq, shifts=1, dims=0) * 0.3
        self.register_buffer("prerequisite_graph", prereq)

    def encode(self, concept_ids: torch.Tensor, mastery: torch.Tensor, confidence: torch.Tensor, uncertainty: torch.Tensor, fatigue: torch.Tensor, retention: torch.Tensor) -> StudentManifoldPoint:
        concept_ids = concept_ids.long().clamp(0, self.num_concepts - 1)
        concept = self.concept_embed(concept_ids)
        x = torch.cat([concept, mastery, confidence, uncertainty, fatigue, retention], dim=-1)
        hidden = self.encoder(x)
        latent = self.to_latent(hidden)
        mastery_logits = self.mastery_head(latent)
        mastery_pred = torch.sigmoid(mastery_logits)
        confidence_pred = clip_prob(torch.sigmoid(self.confidence_head(latent)))
        uncertainty_pred = clip_prob(torch.sigmoid(self.uncertainty_head(latent)))
        curvature = F.softplus(self.curvature_head(latent))
        return StudentManifoldPoint(latent=latent, mastery=mastery_pred, confidence=confidence_pred, uncertainty=uncertainty_pred, curvature=curvature, concept_logits=mastery_logits, metadata={"concept_ids": concept_ids})

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent if latent.ndim > 1 else latent.unsqueeze(0)
        return normalize(self.decoder(latent), dim=-1)

    def decode(self, latent: torch.Tensor) -> dict:
        latent = latent if latent.ndim > 1 else latent.unsqueeze(0)
        return {
            "mastery": torch.sigmoid(self.mastery_head(latent)),
            "confidence": clip_prob(torch.sigmoid(self.confidence_head(latent))),
            "uncertainty": clip_prob(torch.sigmoid(self.uncertainty_head(latent))),
            "curvature": F.softplus(self.curvature_head(latent)),
        }

    def smoothness_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim < 2 or trajectory.size(0) < 2:
            return trajectory.new_tensor(0.0)
        diffs = trajectory[1:] - trajectory[:-1]
        return diffs.pow(2).sum(dim=-1).mean()

    def curvature_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim < 2 or trajectory.size(0) < 3:
            return trajectory.new_tensor(0.0)
        first = trajectory[1:] - trajectory[:-1]
        second = first[1:] - first[:-1]
        return second.pow(2).sum(dim=-1).mean()

    def mastery_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(a, b)

    def manifold_report(self, point: StudentManifoldPoint) -> dict:
        return {
            "latent_norm": float(point.latent.norm(dim=-1).mean().detach().cpu()),
            "mastery_mean": float(point.mastery.mean().detach().cpu()),
            "confidence_mean": float(point.confidence.mean().detach().cpu()),
            "uncertainty_mean": float(point.uncertainty.mean().detach().cpu()),
            "curvature_mean": float(point.curvature.mean().detach().cpu()),
        }
