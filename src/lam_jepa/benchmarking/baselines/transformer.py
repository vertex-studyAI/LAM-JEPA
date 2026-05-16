from __future__ import annotations
import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, num_heads: int = 4, depth: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))
