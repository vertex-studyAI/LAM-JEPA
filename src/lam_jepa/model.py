from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import SparseMemory


@dataclass
class LAMJEPAConfig:
    input_dim: int = 32
    vocab_size: int = 256
    embed_dim: int = 32
    hidden_dim: int = 64
    proj_dim: int = 32
    pred_dim: int = 16
    num_codes: int = 32
    num_actions: int = 8
    num_rubric: int = 4
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    momentum: float = 0.996
    temperature: float = 0.07
    max_steps: int = 3
    memory_size: int = 64
    use_quantizer: bool = True
    use_memory: bool = True
    use_planner: bool = True
    use_target: bool = True

    # additions
    latent_noise_std: float = 0.10
    rollout_samples: int = 1
    use_uncertainty: bool = True
    use_counterfactuals: bool = True


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T, :]
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)


class MultiViewEncoder(nn.Module):
    def __init__(self, cfg: LAMJEPAConfig):
        super().__init__()
        self.token = TokenEncoder(
            cfg.vocab_size, cfg.embed_dim, cfg.hidden_dim, cfg.num_heads, cfg.num_layers, cfg.dropout
        )
        self.numeric = nn.Linear(cfg.input_dim, cfg.embed_dim)
        self.fuse = MLP(cfg.embed_dim * 2, cfg.hidden_dim, cfg.embed_dim, dropout=cfg.dropout)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, numeric_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_token = self.token(tokens)

        if numeric_x is None:
            numeric_x = torch.zeros(tokens.size(0), 1, device=tokens.device, dtype=torch.float32)

        if numeric_x.dim() == 1:
            numeric_x = numeric_x.unsqueeze(-1)

        numeric_x = numeric_x.float()

        if numeric_x.size(-1) < self.numeric.in_features:
            numeric_x = F.pad(numeric_x, (0, self.numeric.in_features - numeric_x.size(-1)))

        numeric_x = numeric_x[..., : self.numeric.in_features]
        z_num = self.numeric(numeric_x)

        z = torch.cat([z_token, z_num], dim=-1)
        return self.norm(self.fuse(z))


class EMAQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_weight", torch.randn(num_codes, dim))

    def forward(self, z: torch.Tensor):
        flat = z.view(-1, self.dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook[indices].view_as(z)

        if self.training:
            one_hot = F.one_hot(indices, self.num_codes).type_as(flat)
            self.ema_count.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)

            n = self.ema_count.sum().clamp_min(self.eps)
            cluster_size = (self.ema_count + self.eps) / (n + self.num_codes * self.eps) * n
            self.codebook.data.copy_(self.ema_weight / cluster_size.unsqueeze(1).clamp_min(self.eps))

        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        quant_loss = commit_loss + codebook_loss

        z_q = z + (z_q - z).detach()
        return z_q, quant_loss, indices


class LatentActionModel(nn.Module):
    """
    Latent policy + stochastic transition.
    This is Mac-safe because it's all standard PyTorch.
    """
    def __init__(self, dim: int, num_actions: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.policy = nn.Linear(dim, num_actions)
        self.action_embed = nn.Embedding(num_actions, dim)

        self.transition_mu = MLP(dim * 4, hidden_dim, dim, dropout=dropout)
        self.transition_logvar = MLP(dim * 4, hidden_dim, dim, dropout=dropout)

        self.norm = nn.LayerNorm(dim)

    def step(
        self,
        z: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        temp: float = 0.7,
        sample: bool = False,
        action_override: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ):
        logits = self.policy(z)
        probs = torch.softmax(logits / max(temp, 1e-6), dim=-1)

        if action_override is None:
            a = probs.multinomial(num_samples=1).squeeze(-1) if sample else probs.argmax(dim=-1)
        else:
            a = action_override.long()

        a_vec = self.action_embed(a)

        if r is None:
            r = torch.zeros_like(z)
        if u is None:
            u = torch.zeros_like(z)

        h = torch.cat([z, a_vec, r, u], dim=-1)

        delta_mu = self.transition_mu(h)
        delta_logvar = self.transition_logvar(h).clamp(-8.0, 4.0)

        if sample:
            eps = torch.randn_like(delta_mu)
            delta = delta_mu + eps * torch.exp(0.5 * delta_logvar)
        else:
            delta = delta_mu

        if noise_std > 0:
            delta = delta + noise_std * torch.randn_like(delta)

        z_next = self.norm(z + delta)
        return z_next, a, logits, probs, delta_mu, delta_logvar

    def rollout(
        self,
        z: torch.Tensor,
        steps: int,
        r: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        temp: float = 0.7,
        sample: bool = False,
        noise_std: float = 0.0,
    ):
        traj: List[torch.Tensor] = [z]
        actions: List[torch.Tensor] = []
        logits_seq: List[torch.Tensor] = []
        delta_mus: List[torch.Tensor] = []
        delta_logvars: List[torch.Tensor] = []

        current = z
        for _ in range(steps):
            current, a, logits, _, mu, logvar = self.step(
                current,
                r=r,
                u=u,
                temp=temp,
                sample=sample,
                noise_std=noise_std,
            )
            traj.append(current)
            actions.append(a)
            logits_seq.append(logits)
            delta_mus.append(mu)
            delta_logvars.append(logvar)

        return traj, actions, logits_seq, delta_mus, delta_logvars

    def counterfactual_rollout(
        self,
        z: torch.Tensor,
        action_id: int,
        steps: int,
        r: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ):
        """
        Roll out a fixed tutoring action through latent dynamics.
        Useful for 'what if we intervene with action k?' experiments.
        """
        traj: List[torch.Tensor] = [z]
        current = z
        action = torch.full(
            (z.size(0),),
            int(action_id),
            device=z.device,
            dtype=torch.long,
        )

        for _ in range(steps):
            current, _, _, _, _, _ = self.step(
                current,
                r=r,
                u=u,
                temp=1.0,
                sample=False,
                action_override=action,
                noise_std=noise_std,
            )
            traj.append(current)

        return traj


class ValueHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = MLP(dim, hidden_dim, 1, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))


class ConfidenceHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = MLP(dim, hidden_dim, 1, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))


class VerifierHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = MLP(dim, hidden_dim, 1, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))


class RubricHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_rubric: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [MLP(dim, hidden_dim, 1, dropout=dropout) for _ in range(num_rubric)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scores = [torch.sigmoid(h(z)) * 8.0 for h in self.heads]
        return torch.cat(scores, dim=-1)


class OutputDecoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = MLP(dim, hidden_dim, vocab_size, dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LAMJEPA(nn.Module):
    def __init__(self, cfg: LAMJEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)

        self.target_encoder = MultiViewEncoder(cfg)
        self.target_projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)

        self.quantizer = EMAQuantizer(cfg.num_codes, cfg.proj_dim)
        self.latent_action = LatentActionModel(cfg.proj_dim, cfg.num_actions, cfg.hidden_dim, cfg.dropout)
        self.memory = SparseMemory(cfg.proj_dim, capacity=cfg.memory_size)

        self.value_head = ValueHead(cfg.proj_dim, cfg.hidden_dim, cfg.dropout)
        self.confidence_head = ConfidenceHead(cfg.proj_dim, cfg.hidden_dim, cfg.dropout)
        self.verifier_head = VerifierHead(cfg.proj_dim, cfg.hidden_dim, cfg.dropout)
        self.rubric_head = RubricHead(cfg.proj_dim, cfg.hidden_dim, cfg.num_rubric, cfg.dropout)
        self.decoder = OutputDecoder(cfg.proj_dim, cfg.vocab_size, cfg.hidden_dim, cfg.dropout)

        self.uncertainty_head = MLP(cfg.proj_dim, cfg.hidden_dim, 1, dropout=cfg.dropout)
        self.latent_summary_head = MLP(cfg.proj_dim, cfg.hidden_dim, cfg.proj_dim, dropout=cfg.dropout)

        self._sync_target()

    @torch.no_grad()
    def _sync_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.copy_(p.data)
        for p, tp in zip(self.projector.parameters(), self.target_projector.parameters()):
            tp.data.copy_(p.data)

    @torch.no_grad()
    def update_target(self):
        tau = self.cfg.momentum
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)
        for p, tp in zip(self.projector.parameters(), self.target_projector.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def forward(
        self,
        tokens: torch.Tensor,
        numeric_x: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        temp: Optional[float] = None,
        sample_rollout: Optional[bool] = None,
        noise_std: Optional[float] = None,
    ):
        if steps is None:
            steps = self.cfg.max_steps
        if temp is None:
            temp = self.cfg.temperature
        if sample_rollout is None:
            sample_rollout = self.training
        if noise_std is None:
            noise_std = self.cfg.latent_noise_std if sample_rollout else 0.0

        z = self.encoder(tokens, numeric_x=numeric_x)
        z = self.projector(z)

        if self.cfg.use_quantizer:
            z_q, quant_loss, indices = self.quantizer(z)
        else:
            z_q = z
            quant_loss = z.new_tensor(0.0)
            indices = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        if self.cfg.use_memory:
            r = self.memory.retrieve(z_q)
            z_mem = self.memory.gated_correction(z_q, r)
        else:
            r = torch.zeros_like(z_q)
            z_mem = z_q

        if self.cfg.use_planner and steps > 0:
            traj, actions, logits_seq, delta_mus, delta_logvars = self.latent_action.rollout(
                z_mem,
                steps=steps,
                r=r,
                u=None,
                temp=temp,
                sample=sample_rollout,
                noise_std=noise_std,
            )
        else:
            traj, actions, logits_seq, delta_mus, delta_logvars = [z_mem], [], [], [], []

        final = traj[-1]
        latent_summary = self.latent_summary_head(final)

        logits = self.decoder(final)
        value = self.value_head(final)
        confidence = self.confidence_head(final)
        verifier = self.verifier_head(final)
        rubric = self.rubric_head(final)

        if self.cfg.use_uncertainty:
            uncertainty = F.softplus(self.uncertainty_head(final)) + 1e-6
        else:
            uncertainty = torch.zeros(final.size(0), 1, device=final.device, dtype=final.dtype)

        if self.cfg.use_target:
            with torch.no_grad():
                t_z = self.target_encoder(tokens, numeric_x=numeric_x)
                t_z = self.target_projector(t_z)
        else:
            t_z = z.detach()

        return {
            "z": z,
            "z_q": z_q,
            "target_z": t_z,
            "traj": traj,
            "actions": actions,
            "action_logits": logits_seq,
            "delta_mus": delta_mus,
            "delta_logvars": delta_logvars,
            "logits": logits,
            "value": value,
            "confidence": confidence,
            "verifier": verifier,
            "rubric": rubric,
            "uncertainty": uncertainty,
            "latent_summary": latent_summary,
            "quant_loss": quant_loss,
            "indices": indices,
            "memory_read": r,
            "z_mem": z_mem,
        }

    @torch.no_grad()
    def predict(self, tokens: torch.Tensor, numeric_x: Optional[torch.Tensor] = None, steps: int = 0):
        out = self.forward(tokens, numeric_x=numeric_x, steps=steps, sample_rollout=False)
        probs = torch.softmax(out["logits"], dim=-1)
        return {
            **out,
            "probs": probs,
            "pred": probs.argmax(dim=-1),
        }

    @torch.no_grad()
    def imagine_counterfactuals(
        self,
        tokens: torch.Tensor,
        numeric_x: Optional[torch.Tensor] = None,
        steps: int = 3,
        action_ids: Optional[List[int]] = None,
    ):
        """
        Returns several latent futures under different fixed tutoring actions.
        Good for proving intervention sensitivity.
        """
        out = self.forward(tokens, numeric_x=numeric_x, steps=0)
        z0 = out["z_mem"]

        if action_ids is None:
            action_ids = list(range(self.cfg.num_actions))

        futures = {}
        for aid in action_ids:
            futures[int(aid)] = self.latent_action.counterfactual_rollout(
                z0,
                action_id=int(aid),
                steps=steps,
                r=out["memory_read"],
                u=None,
                noise_std=0.0,
            )

        return {
            "base": out,
            "counterfactual_futures": futures,
        }