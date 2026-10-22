"""EMAQuantizer class transcribed without scientific changes from frozen model.py.

Source: vertex-studyAI/LAM-JEPA@760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb
Path: src/lam_jepa/model.py
Full source blob: a98e1c6b5aaf97979b73fd0f8e623482a341d7bd
Only imports needed by this class are retained. This is an extracted source
component for a first-forward synthetic implementation check, not an ARC model
or a replacement for the original scientific experiment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
