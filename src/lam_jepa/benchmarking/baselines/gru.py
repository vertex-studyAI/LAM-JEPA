from __future__ import annotations
import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        _, h = self.gru(x)
        return self.head(h[-1])
