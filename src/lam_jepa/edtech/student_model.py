from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.common import build_mlp, clip_prob


@dataclass
class StudentInteraction:
    concept_id: int
    correct: float
    confidence: float = 0.5
    difficulty: float = 0.5
    response_time: float = 1.0
    misconception_id: int = -1
    explanation_type: str = "guided_hint"
    intervention: str = "generic"
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class StudentState:
    mastery: torch.Tensor
    confidence: torch.Tensor
    learning_velocity: torch.Tensor
    misconception_logits: torch.Tensor
    knowledge_graph: torch.Tensor
    uncertainty: torch.Tensor
    fatigue: torch.Tensor
    retention: torch.Tensor
    emotional_state: torch.Tensor
    hidden: torch.Tensor
    latent_state: torch.Tensor

    def as_dict(self) -> dict:
        return {
            "mastery": self.mastery,
            "confidence": self.confidence,
            "learning_velocity": self.learning_velocity,
            "misconception_logits": self.misconception_logits,
            "knowledge_graph": self.knowledge_graph,
            "uncertainty": self.uncertainty,
            "fatigue": self.fatigue,
            "retention": self.retention,
            "emotional_state": self.emotional_state,
            "hidden": self.hidden,
            "latent_state": self.latent_state,
        }


class StudentModel(nn.Module):
    """Latent student digital twin with mastery, confidence, retention and fatigue estimates."""

    def __init__(
        self,
        num_concepts: int = 64,
        num_misconceptions: int = 24,
        hidden_dim: int = 160,
        graph_temperature: float = 0.7,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_misconceptions = num_misconceptions
        self.hidden_dim = hidden_dim
        self.graph_temperature = graph_temperature

        self.concept_embed = nn.Embedding(num_concepts, hidden_dim)
        self.correct_embed = nn.Embedding(2, hidden_dim)
        self.confidence_proj = nn.Linear(1, hidden_dim)
        self.difficulty_proj = nn.Linear(1, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.response_proj = nn.Linear(1, hidden_dim)
        self.intervention_proj = build_mlp(2, hidden_dim, hidden_dim, depth=2)

        self.gru = nn.GRU(hidden_dim * 7, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self.mastery_head = build_mlp(hidden_dim, hidden_dim, num_concepts, depth=3)
        self.confidence_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.velocity_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.misconception_head = build_mlp(hidden_dim, hidden_dim, num_misconceptions, depth=3)
        self.graph_head = build_mlp(hidden_dim, hidden_dim, num_concepts * num_concepts, depth=2)
        self.uncertainty_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.fatigue_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.retention_head = build_mlp(hidden_dim, hidden_dim, 1, depth=2)
        self.emotion_head = build_mlp(hidden_dim, hidden_dim, 3, depth=2)
        self.latent_head = build_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

    def _sequence_features(self, interactions: Sequence[StudentInteraction], device: torch.device) -> torch.Tensor:
        if len(interactions) == 0:
            return torch.zeros(1, 1, self.hidden_dim * 7, device=device)
        concept = torch.tensor([i.concept_id for i in interactions], device=device).clamp(0, self.num_concepts - 1)
        correct = torch.tensor([1 if i.correct >= 0.5 else 0 for i in interactions], device=device)
        confidence = torch.tensor([[float(i.confidence)] for i in interactions], device=device)
        difficulty = torch.tensor([[float(i.difficulty)] for i in interactions], device=device)
        response_time = torch.tensor([[float(i.response_time)] for i in interactions], device=device)
        timestamp = torch.tensor([[float(i.timestamp)] for i in interactions], device=device)
        intervention = torch.tensor([[float(hash(i.intervention) % 997) / 997.0, float(hash(i.explanation_type) % 997) / 997.0] for i in interactions], device=device)
        x = torch.cat(
            [
                self.concept_embed(concept),
                self.correct_embed(correct),
                self.confidence_proj(confidence),
                self.difficulty_proj(difficulty),
                self.time_proj(timestamp),
                self.response_proj(response_time),
                self.intervention_proj(intervention),
            ],
            dim=-1,
        ).unsqueeze(0)
        return x

    def _from_tensor(self, tensor: torch.Tensor) -> StudentState:
        if tensor.ndim == 2:
            x = tensor.unsqueeze(1)
        elif tensor.ndim == 3:
            x = tensor
        else:
            raise ValueError("tensor inputs must be 2D or 3D")
        out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        return self._state_from_hidden(h)

    def _state_from_hidden(self, h: torch.Tensor) -> StudentState:
        mastery_logits = self.mastery_head(h)
        mastery = torch.sigmoid(mastery_logits)
        confidence = clip_prob(torch.sigmoid(self.confidence_head(h)))
        learning_velocity = F.softplus(self.velocity_head(h))
        misconception_logits = self.misconception_head(h)
        graph_logits = self.graph_head(h).view(-1, self.num_concepts, self.num_concepts)
        knowledge_graph = torch.softmax(graph_logits / self.graph_temperature, dim=-1)
        uncertainty = clip_prob(torch.sigmoid(self.uncertainty_head(h)))
        fatigue = clip_prob(torch.sigmoid(self.fatigue_head(h)))
        retention = clip_prob(torch.sigmoid(self.retention_head(h)))
        emotional_state = torch.softmax(self.emotion_head(h), dim=-1)
        latent_state = self.latent_head(h)
        return StudentState(
            mastery=mastery.squeeze(0),
            confidence=confidence.squeeze(0),
            learning_velocity=learning_velocity.squeeze(0),
            misconception_logits=misconception_logits.squeeze(0),
            knowledge_graph=knowledge_graph.squeeze(0),
            uncertainty=uncertainty.squeeze(0),
            fatigue=fatigue.squeeze(0),
            retention=retention.squeeze(0),
            emotional_state=emotional_state.squeeze(0),
            hidden=h.squeeze(0),
            latent_state=latent_state.squeeze(0),
        )

    def forward(self, interactions: Sequence[StudentInteraction] | torch.Tensor, hidden: Optional[torch.Tensor] = None) -> StudentState:
        device = next(self.parameters()).device
        if isinstance(interactions, torch.Tensor):
            return self._from_tensor(interactions.to(device))
        x = self._sequence_features(interactions, device=device)
        if hidden is not None:
            out, _ = self.gru(x, hidden)
        else:
            out, _ = self.gru(x)
        h = self.norm(out[:, -1, :])
        return self._state_from_hidden(h)

    @torch.no_grad()
    def update_state(self, interactions: Sequence[StudentInteraction], hidden: Optional[torch.Tensor] = None) -> StudentState:
        self.eval()
        return self.forward(interactions, hidden=hidden)

    @torch.no_grad()
    def summarize_state(self, state: StudentState) -> dict:
        return {
            "mean_mastery": float(state.mastery.mean().item()),
            "mean_confidence": float(state.confidence.mean().item()),
            "mean_velocity": float(state.learning_velocity.mean().item()),
            "mean_uncertainty": float(state.uncertainty.mean().item()),
            "mean_fatigue": float(state.fatigue.mean().item()),
            "mean_retention": float(state.retention.mean().item()),
            "emotion_entropy": float(-(state.emotional_state * (state.emotional_state + 1e-8).log()).sum().item()),
        }

    @torch.no_grad()
    def predict_next_correctness(self, state: StudentState, concept_id: int) -> float:
        idx = int(max(0, min(self.num_concepts - 1, concept_id)))
        mastery = float(state.mastery[idx].item())
        confidence = float(state.confidence.mean().item())
        retention = float(state.retention.mean().item())
        fatigue = float(state.fatigue.mean().item())
        probability = 0.5 * mastery + 0.2 * confidence + 0.2 * retention - 0.2 * fatigue
        return float(max(0.0, min(1.0, probability)))
