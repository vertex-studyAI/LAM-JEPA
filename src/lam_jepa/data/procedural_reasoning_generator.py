from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import random
import torch

from . import Batch, text_to_tokens


@dataclass
class ReasoningExample:
    prompt: str
    solution: str
    concept: str
    difficulty: float
    rubric: list[float]
    metadata: dict = field(default_factory=dict)

    def to_batch(self, vocab_size: int = 256) -> Batch:
        tokens = text_to_tokens(f"{self.prompt} {self.solution}", vocab_size=vocab_size)
        numeric_x = torch.tensor([[float(self.metadata.get("concept_id", 0)), float(self.difficulty), float(len(self.prompt.split())), float(len(self.solution.split()))]], dtype=torch.float32)
        labels = torch.tensor([int(self.metadata.get("label", 0))], dtype=torch.long)
        return Batch(tokens=tokens.unsqueeze(0), numeric_x=numeric_x, labels=labels, rubric=torch.tensor([self.rubric], dtype=torch.float32), prompt=self.prompt, solution=self.solution, concept=self.concept, difficulty=self.difficulty, metadata=dict(self.metadata))


def generate_algebra_problem(difficulty: float = 0.5, rng: random.Random | None = None) -> ReasoningExample:
    rng = rng or random.Random()
    x = rng.randint(1, 12)
    a = rng.randint(1, 8)
    b = rng.randint(0, 12)
    c = a * x + b
    prompt = f"Solve for x: {a}x + {b} = {c}"
    solution = f"x = {x}; subtract {b} then divide by {a}."
    return ReasoningExample(prompt=prompt, solution=solution, concept="algebra_equation", difficulty=float(difficulty), rubric=[8.0, float(a), float(b), 7.0], metadata={"concept_id": 1, "label": x})


def generate_physics_problem(difficulty: float = 0.6, rng: random.Random | None = None) -> ReasoningExample:
    rng = rng or random.Random()
    mass = rng.randint(1, 8)
    acc = rng.randint(1, 6)
    force = mass * acc
    prompt = f"A {mass} kg object accelerates at {acc} m/s^2. What net force acts on it?"
    solution = f"{force} N by F=ma."
    return ReasoningExample(prompt=prompt, solution=solution, concept="physics_force", difficulty=float(difficulty), rubric=[float(force % 8), 8.0, 7.0, 6.0], metadata={"concept_id": 10, "label": force})


def generate_science_problem(topic: str = "thermodynamics", difficulty: float = 0.55, rng: random.Random | None = None) -> ReasoningExample:
    rng = rng or random.Random()
    if topic == "electricity":
        current = rng.randint(1, 5)
        resistance = rng.randint(1, 8)
        voltage = current * resistance
        prompt = f"What voltage is needed for {current} A through a {resistance} ohm resistor?"
        solution = f"{voltage} V using V=IR."
        concept = "science_electricity"
        rubric = [float(voltage % 8), 8.0, 7.0, 6.0]
        label = voltage
    else:
        prompt = "Why does a gas expand when heated at constant pressure?"
        solution = "Particles move faster and occupy more volume."
        concept = "science_thermodynamics"
        rubric = [7.0, 6.0, 8.0, 7.0]
        label = 0
    return ReasoningExample(prompt=prompt, solution=solution, concept=concept, difficulty=float(difficulty), rubric=rubric, metadata={"label": label})


def generate_reasoning_problem(kind: str = "logic", difficulty: float = 0.5, rng: random.Random | None = None) -> ReasoningExample:
    rng = rng or random.Random()
    if kind == "logic":
        prompt = "All engineers like puzzles. Maya likes puzzles. Can we conclude Maya is an engineer?"
        solution = "No; liking puzzles is not sufficient evidence."
        concept = "logic"
        rubric = [8.0, 6.0, 7.0, 8.0]
    elif kind == "sequence":
        a = [2, 6, 12, 20]
        prompt = f"What comes next: {', '.join(map(str, a))}, ?"
        solution = "30; the differences grow by two each step."
        concept = "sequence_reasoning"
        rubric = [6.0, 8.0, 7.0, 6.0]
    else:
        prompt = "A claim is supported by one study but not another. What cautious conclusion is best?"
        solution = "The claim is context-dependent and needs more evidence."
        concept = "scientific_reasoning"
        rubric = [7.0, 7.0, 8.0, 7.0]
    return ReasoningExample(prompt=prompt, solution=solution, concept=concept, difficulty=float(difficulty), rubric=rubric, metadata={"label": 0})


def generate_reasoning_curriculum(num_items: int = 32, seed: int = 7) -> list[ReasoningExample]:
    rng = random.Random(seed)
    curriculum: list[ReasoningExample] = []
    for i in range(num_items):
        p = i / max(num_items - 1, 1)
        if p < 0.3:
            curriculum.append(generate_algebra_problem(difficulty=0.2 + 0.4 * p, rng=rng))
        elif p < 0.6:
            curriculum.append(generate_physics_problem(difficulty=0.45 + 0.3 * p, rng=rng))
        elif p < 0.8:
            curriculum.append(generate_science_problem("electricity" if i % 2 else "thermodynamics", difficulty=0.55 + 0.25 * p, rng=rng))
        else:
            curriculum.append(generate_reasoning_problem("logic" if i % 2 else "sequence", difficulty=0.65 + 0.25 * p, rng=rng))
    return curriculum
