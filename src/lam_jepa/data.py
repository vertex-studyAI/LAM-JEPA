from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Dict, Optional

import random
import math

import torch


@dataclass
class Batch:
    tokens: torch.Tensor
    numeric_x: torch.Tensor
    labels: torch.Tensor
    rubric: torch.Tensor
    prompt: str | None = None
    solution: str | None = None
    concept: str | None = None
    difficulty: float | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


def _make_tokens(batch: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq_len))


def _hash_token(text: str, vocab_size: int) -> int:
    h = blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % vocab_size


def text_to_tokens(text: str, vocab_size: int = 256, max_len: int = 24) -> torch.Tensor:
    words = text.lower().split()
    ids = [_hash_token(w, vocab_size) for w in words[:max_len]]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def modular_addition_batch(batch: int = 64, mod: int = 16, seq_len: int = 12, vocab_size: int = 256) -> Batch:
    a = torch.randint(0, mod, (batch,))
    b = torch.randint(0, mod, (batch,))
    y = (a + b) % mod
    tokens = _make_tokens(batch, seq_len, vocab_size)
    numeric_x = torch.stack([a.float(), b.float()], dim=-1)
    rubric = torch.stack([
        (y % 8).float(),
        ((a < b).float() * 8.0),
        (((a + b) > mod).float() * 8.0),
        torch.ones_like(y).float() * 6.0,
    ], dim=-1)
    return Batch(tokens=tokens, numeric_x=numeric_x, labels=y, rubric=rubric, concept="modular_addition", difficulty=0.25)


def chained_arithmetic_batch(batch: int = 64, mod: int = 16, seq_len: int = 16, vocab_size: int = 256) -> Batch:
    a = torch.randint(1, 100, (batch,))
    b = torch.randint(1, 100, (batch,))
    c = torch.randint(1, 100, (batch,))
    y = (a + b - c) % mod
    tokens = _make_tokens(batch, seq_len, vocab_size)
    numeric_x = torch.stack([a.float(), b.float(), c.float()], dim=-1)
    rubric = torch.stack([
        torch.clamp((y % 4).float() * 2.0, 0, 8),
        torch.clamp((a > b).float() * 8.0, 0, 8),
        torch.ones_like(y).float() * 7.0,
        torch.clamp((c < a + b).float() * 8.0, 0, 8),
    ], dim=-1)
    return Batch(tokens=tokens, numeric_x=numeric_x, labels=y.long(), rubric=rubric, concept="chained_arithmetic", difficulty=0.6)


def parity_batch(batch: int = 64, seq_len: int = 8, vocab_size: int = 256) -> Batch:
    x = torch.randint(0, 2, (batch, 32))
    y = x.sum(dim=-1) % 2
    tokens = _make_tokens(batch, seq_len, vocab_size)
    numeric_x = x.float()
    rubric = torch.stack([
        (1 - y).float() * 8.0,
        y.float() * 8.0,
        torch.ones_like(y).float() * 7.0,
        torch.ones_like(y).float() * 6.0,
    ], dim=-1)
    return Batch(tokens=tokens, numeric_x=numeric_x, labels=y, rubric=rubric, concept="parity", difficulty=0.15)


def algebra_batch(batch: int = 64, mod: int = 16, seq_len: int = 14, vocab_size: int = 256) -> Batch:
    x = torch.randint(0, mod, (batch,))
    a = torch.randint(1, 8, (batch,))
    b = torch.randint(0, 8, (batch,))
    c = a * x + b
    y = x % mod
    tokens = _make_tokens(batch, seq_len, vocab_size)
    numeric_x = torch.stack([a.float(), b.float(), c.float()], dim=-1)
    rubric = torch.stack([
        (y % 8).float(),
        (a.float() / 8.0) * 8.0,
        (b.float() / 8.0) * 8.0,
        torch.ones_like(y).float() * 6.0,
    ], dim=-1)
    return Batch(tokens=tokens, numeric_x=numeric_x, labels=y.long(), rubric=rubric, concept="algebra", difficulty=0.45)


def sample_batch(task: str, batch: int = 64, vocab_size: int = 256) -> Batch:
    if task == "modadd":
        return modular_addition_batch(batch=batch, vocab_size=vocab_size)
    if task == "chain":
        return chained_arithmetic_batch(batch=batch, vocab_size=vocab_size)
    if task == "parity":
        return parity_batch(batch=batch, vocab_size=vocab_size)
    if task == "algebra":
        return algebra_batch(batch=batch, vocab_size=vocab_size)
    if task == "gsm8k":
        from .edtech.tasks import generate_math_example
        ex = generate_math_example("gsm8k", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    if task == "equation":
        from .edtech.tasks import generate_math_example
        ex = generate_math_example("equation", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    if task == "science":
        from .edtech.tasks import generate_science_example
        ex = generate_science_example("physics", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    if task == "reading":
        from .edtech.tasks import generate_reading_example
        ex = generate_reading_example("comprehension", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    if task == "tutoring":
        from .edtech.tasks import generate_tutoring_example
        ex = generate_tutoring_example("fractions", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    if task == "reasoning":
        from .edtech.tasks import generate_reasoning_example
        ex = generate_reasoning_example("logic", vocab_size=vocab_size)
        return Batch(
            tokens=ex.tokens.repeat(batch, 1),
            numeric_x=ex.numeric_x.repeat(batch, 1),
            labels=ex.labels.repeat(batch),
            rubric=ex.rubric.repeat(batch, 1),
            prompt=ex.prompt,
            solution=ex.solution,
            concept=ex.concept,
            difficulty=ex.difficulty,
            metadata=ex.metadata,
        )
    raise ValueError(f"unknown task: {task}")


class Curriculum:
    def __init__(self):
        self.level = 0
        self.tasks = ["parity", "modadd", "algebra", "chain", "gsm8k", "science", "reading", "tutoring", "reasoning"]

    def update(self, accuracy: float) -> None:
        if accuracy > 0.85 and self.level < len(self.tasks) - 1:
            self.level += 1

    def sample(self) -> str:
        return self.tasks[self.level]
