from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Dict, Optional

import math
import random

import torch


SUPPORTED_TASKS = (
    "parity",
    "modadd",
    "algebra",
    "chain",
    "gsm8k",
    "equation",
    "science",
    "reading",
    "tutoring",
    "reasoning",
)


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


@dataclass
class Curriculum:
    level: int = 0
    tasks: list[str] = field(default_factory=lambda: list(SUPPORTED_TASKS))

    def update(self, accuracy: float) -> None:
        if accuracy > 0.85 and self.level < len(self.tasks) - 1:
            self.level += 1
        elif accuracy < 0.45 and self.level > 0:
            self.level -= 1

    def sample(self) -> str:
        return self.tasks[self.level]


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


def _pack(prompt: str, answer: str, concept_id: int, difficulty: float, rubric: list[float], vocab_size: int = 256) -> Batch:
    tokens = text_to_tokens(f"{prompt} {answer}", vocab_size=vocab_size)
    numeric_x = torch.tensor([[float(concept_id), float(difficulty), float(len(prompt.split())), float(len(answer.split()))]], dtype=torch.float32)
    labels = torch.tensor([concept_id % vocab_size], dtype=torch.long)
    rubric_t = torch.tensor([rubric], dtype=torch.float32)
    return Batch(tokens=tokens.unsqueeze(0), numeric_x=numeric_x, labels=labels, rubric=rubric_t, prompt=prompt, solution=answer, concept=f"concept_{concept_id}", difficulty=difficulty, metadata={"prompt": prompt, "answer": answer})


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


def equation_batch(batch: int = 64, mod: int = 16, seq_len: int = 18, vocab_size: int = 256) -> Batch:
    """Generate one-step distributive equations of the form a(x + b) = c."""
    x = torch.randint(0, mod, (batch,))
    a = torch.randint(1, 8, (batch,))
    b = torch.randint(0, 8, (batch,))
    c = a * (x + b)
    y = x % mod
    tokens = _make_tokens(batch, seq_len, vocab_size)
    numeric_x = torch.stack([a.float(), b.float(), c.float()], dim=-1)
    rubric = torch.stack([
        (y % 8).float(),
        torch.ones_like(y).float() * 8.0,
        torch.clamp((a + b).float(), 0, 8),
        torch.ones_like(y).float() * 7.0,
    ], dim=-1)
    return Batch(
        tokens=tokens,
        numeric_x=numeric_x,
        labels=y.long(),
        rubric=rubric,
        concept="distributive_equation",
        difficulty=0.58,
    )


def _repeat_example(example: Batch, batch: int) -> Batch:
    if batch <= 1:
        return example
    tokens = example.tokens.repeat(batch, 1)
    numeric_x = example.numeric_x.repeat(batch, 1)
    labels = example.labels.repeat(batch)
    rubric = example.rubric.repeat(batch, 1)
    return Batch(tokens=tokens, numeric_x=numeric_x, labels=labels, rubric=rubric, prompt=example.prompt, solution=example.solution, concept=example.concept, difficulty=example.difficulty, metadata=dict(example.metadata))


def _science_batch(kind: str, batch: int, vocab_size: int) -> Batch:
    if kind == "force":
        mass = torch.randint(1, 8, (batch,))
        acc = torch.randint(1, 6, (batch,))
        force = mass * acc
        tokens = _make_tokens(batch, 14, vocab_size)
        numeric_x = torch.stack([mass.float(), acc.float()], dim=-1)
        rubric = torch.stack([force.float() % 8, torch.ones_like(force).float() * 8.0, torch.ones_like(force).float() * 7.0, torch.ones_like(force).float() * 6.0], dim=-1)
        return Batch(tokens=tokens, numeric_x=numeric_x, labels=(force % vocab_size).long(), rubric=rubric, concept="science_force", difficulty=0.55)
    if kind == "electricity":
        current = torch.randint(1, 5, (batch,))
        resistance = torch.randint(1, 8, (batch,))
        voltage = current * resistance
        tokens = _make_tokens(batch, 14, vocab_size)
        numeric_x = torch.stack([current.float(), resistance.float()], dim=-1)
        rubric = torch.stack([voltage.float() % 8, torch.ones_like(voltage).float() * 8.0, torch.ones_like(voltage).float() * 7.0, torch.ones_like(voltage).float() * 6.0], dim=-1)
        return Batch(tokens=tokens, numeric_x=numeric_x, labels=(voltage % vocab_size).long(), rubric=rubric, concept="science_electricity", difficulty=0.6)
    if kind == "thermo":
        tokens = _make_tokens(batch, 16, vocab_size)
        numeric_x = torch.randn(batch, 2)
        labels = torch.zeros(batch, dtype=torch.long)
        rubric = torch.stack([torch.ones(batch) * 7.0, torch.ones(batch) * 6.0, torch.ones(batch) * 8.0, torch.ones(batch) * 7.0], dim=-1)
        return Batch(tokens=tokens, numeric_x=numeric_x, labels=labels, rubric=rubric, concept="science_thermo", difficulty=0.62)
    return Batch(tokens=_make_tokens(batch, 16, vocab_size), numeric_x=torch.randn(batch, 2), labels=torch.zeros(batch, dtype=torch.long), rubric=torch.ones(batch, 4) * 6.0, concept="science", difficulty=0.5)


def _reading_batch(kind: str, batch: int, vocab_size: int) -> Batch:
    prompt = "Read the passage and select the best answer." if kind == "comprehension" else "Choose the best evidence sentence."
    if kind == "main_idea":
        prompt = "Read the passage and choose the main idea: The article explains how coral reefs support marine life."
        answer = "Coral reefs provide habitats and ecosystem support."
        concept_id = 20
    elif kind == "evidence":
        prompt = "Which sentence best supports the claim that the character is brave?"
        answer = "The character enters the storm to save the child."
        concept_id = 21
    else:
        prompt = "What can be inferred about the speaker's mood from the poem?"
        answer = "The speaker feels reflective and slightly hopeful."
        concept_id = 22
    return _repeat_example(_pack(prompt, answer, concept_id, 0.5, [8.0, 7.0, 6.0, 7.0], vocab_size), batch)


def _tutoring_batch(kind: str, batch: int, vocab_size: int) -> Batch:
    if kind == "fractions":
        prompt = "A student thinks 1/3 is larger than 1/2 because 3 is bigger than 2. Diagnose the misconception."
        answer = "Denominator confusion."
        concept_id = 30
    elif kind == "signs":
        prompt = "A learner changes +5 to -5 when moving it across the equals sign. What error occurred?"
        answer = "Sign error / inverse operation error."
        concept_id = 31
    else:
        prompt = "A student says the same explanation works for every problem. What coaching response helps?"
        answer = "Use a counterexample and ask them to compare cases."
        concept_id = 32
    return _repeat_example(_pack(prompt, answer, concept_id, 0.55, [8.0, 7.0, 7.0, 6.0], vocab_size), batch)


def _reasoning_batch(kind: str, batch: int, vocab_size: int) -> Batch:
    if kind == "logic":
        prompt = "All engineers like puzzles. Maya likes puzzles. Can we conclude Maya is an engineer?"
        answer = "No, the premise is not sufficient."
        concept_id = 40
    elif kind == "sequence":
        prompt = "What comes next: 2, 6, 12, 20, ?"
        answer = "30."
        concept_id = 41
    else:
        prompt = "A claim is supported by data in one experiment but not another. What is the cautious conclusion?"
        answer = "The claim is context-dependent and needs more evidence."
        concept_id = 42
    return _repeat_example(_pack(prompt, answer, concept_id, 0.65, [7.0, 7.0, 8.0, 7.0], vocab_size), batch)


def sample_batch(task: str, batch: int = 64, vocab_size: int = 256) -> Batch:
    if task == "modadd":
        return modular_addition_batch(batch=batch, vocab_size=vocab_size)
    if task == "chain":
        return chained_arithmetic_batch(batch=batch, vocab_size=vocab_size)
    if task == "parity":
        return parity_batch(batch=batch, vocab_size=vocab_size)
    if task == "algebra":
        return algebra_batch(batch=batch, vocab_size=vocab_size)
    if task == "equation":
        return equation_batch(batch=batch, vocab_size=vocab_size)
    if task == "gsm8k":
        return _repeat_example(_pack("A student buys 4 packs with 3 pencils each and gives away 2 pencils. How many pencils remain?", "10. Multiply packs by pencils and subtract the pencils given away.", 0, 0.65, [2.0, 8.0, 7.0, 6.0], vocab_size), batch)
    if task == "science":
        return _science_batch("force", batch=batch, vocab_size=vocab_size)
    if task == "reading":
        return _reading_batch("main_idea", batch=batch, vocab_size=vocab_size)
    if task == "tutoring":
        return _tutoring_batch("fractions", batch=batch, vocab_size=vocab_size)
    if task == "reasoning":
        return _reasoning_batch("logic", batch=batch, vocab_size=vocab_size)
    raise ValueError(f"unknown task: {task}")


from .procedural_reasoning_generator import (  # noqa: E402
    ReasoningExample,
    generate_algebra_problem,
    generate_physics_problem,
    generate_reasoning_curriculum,
    generate_reasoning_problem,
    generate_science_problem,
)
from .student_simulator import StudentSimulator, StudentTrace  # noqa: E402

__all__ = [
    "Batch",
    "Curriculum",
    "ReasoningExample",
    "SUPPORTED_TASKS",
    "StudentSimulator",
    "StudentTrace",
    "equation_batch",
    "sample_batch",
    "text_to_tokens",
    "generate_algebra_problem",
    "generate_physics_problem",
    "generate_reasoning_curriculum",
    "generate_reasoning_problem",
    "generate_science_problem",
]
