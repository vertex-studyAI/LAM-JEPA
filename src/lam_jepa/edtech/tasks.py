from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Optional

import torch

from ..data import Batch


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


def generate_math_example(kind: str = "algebra", difficulty: float = 0.5, vocab_size: int = 256) -> Batch:
    if kind == "gsm8k":
        a = torch.randint(2, 10, (1,)).item()
        b = torch.randint(2, 10, (1,)).item()
        c = torch.randint(1, 8, (1,)).item()
        prompt = f"A student buys {a} packs with {b} pencils each and gives away {c} pencils. How many pencils remain?"
        ans = a * b - c
        steps = [
            f"Multiply packs by pencils: {a} × {b} = {a*b}.",
            f"Subtract the pencils given away: {a*b} - {c} = {ans}.",
        ]
        concept = 0
        rubric = [float(ans % 8), 8.0 if ans >= 0 else 2.0, 7.0, 6.0]
        return _pack(prompt, f"{ans}. {' '.join(steps)}", concept, difficulty, rubric, vocab_size)
    if kind == "equation":
        x = torch.randint(0, 20, (1,)).item()
        a = torch.randint(1, 8, (1,)).item()
        b = torch.randint(1, 12, (1,)).item()
        c = a * x + b
        prompt = f"Solve for x: {a}x + {b} = {c}"
        ans = x
        steps = [
            f"Subtract {b} from both sides.",
            f"Divide by {a}.",
            f"x = {ans}.",
        ]
        rubric = [float(ans % 8), float(a), float(b), 8.0]
        return _pack(prompt, f"x = {ans}. {' '.join(steps)}", 1, difficulty, rubric, vocab_size)
    if kind == "symbolic":
        x = torch.randint(2, 8, (1,)).item()
        prompt = f"Simplify: (x + {x}) + (2x - {x})"
        ans = f"3x"
        steps = ["Combine like terms.", f"The result is {ans}."]
        rubric = [3.0, 7.0, 6.0, 8.0]
        return _pack(prompt, f"{ans}. {' '.join(steps)}", 2, difficulty, rubric, vocab_size)
    prompt = "Solve the linear equation 2x + 5 = 13."
    ans = "x = 4."
    rubric = [4.0, 8.0, 5.0, 7.0]
    return _pack(prompt, ans, 3, difficulty, rubric, vocab_size)


def generate_science_example(kind: str = "physics", difficulty: float = 0.5, vocab_size: int = 256) -> Batch:
    if kind == "force":
        mass = torch.randint(1, 8, (1,)).item()
        acc = torch.randint(1, 6, (1,)).item()
        force = mass * acc
        prompt = f"A {mass} kg object accelerates at {acc} m/s^2. What net force acts on it?"
        ans = f"{force} N."
        rubric = [float(force % 8), 8.0, 7.0, 6.0]
        return _pack(prompt, ans, 10, difficulty, rubric, vocab_size)
    if kind == "thermo":
        prompt = "Why does a gas expand when heated at constant pressure?"
        ans = "Particles move faster and occupy more volume."
        rubric = [7.0, 6.0, 8.0, 7.0]
        return _pack(prompt, ans, 11, difficulty, rubric, vocab_size)
    if kind == "electricity":
        current = torch.randint(1, 5, (1,)).item()
        resistance = torch.randint(1, 8, (1,)).item()
        voltage = current * resistance
        prompt = f"What voltage is needed for {current} A through a {resistance} ohm resistor?"
        ans = f"{voltage} V."
        rubric = [float(voltage % 8), 8.0, 7.0, 6.0]
        return _pack(prompt, ans, 12, difficulty, rubric, vocab_size)
    prompt = "A runner speeds up from rest to 10 m/s in 5 s. What is the acceleration?"
    ans = "2 m/s^2."
    rubric = [2.0, 7.0, 6.0, 8.0]
    return _pack(prompt, ans, 13, difficulty, rubric, vocab_size)


def generate_reading_example(kind: str = "comprehension", difficulty: float = 0.5, vocab_size: int = 256) -> Batch:
    if kind == "main_idea":
        prompt = "Read the passage and choose the main idea: The article explains how coral reefs support marine life."
        ans = "Coral reefs provide habitats and ecosystem support."
        rubric = [8.0, 7.0, 6.0, 7.0]
        return _pack(prompt, ans, 20, difficulty, rubric, vocab_size)
    if kind == "evidence":
        prompt = "Which sentence best supports the claim that the character is brave?"
        ans = "The character enters the storm to save the child."
        rubric = [7.0, 8.0, 6.0, 7.0]
        return _pack(prompt, ans, 21, difficulty, rubric, vocab_size)
    prompt = "What can be inferred about the speaker's mood from the poem?"
    ans = "The speaker feels reflective and slightly hopeful."
    rubric = [7.0, 6.0, 8.0, 7.0]
    return _pack(prompt, ans, 22, difficulty, rubric, vocab_size)


def generate_tutoring_example(kind: str = "misconception", difficulty: float = 0.5, vocab_size: int = 256) -> Batch:
    if kind == "fractions":
        prompt = "A student thinks 1/3 is larger than 1/2 because 3 is bigger than 2. Diagnose the misconception."
        ans = "Denominator confusion."
        rubric = [8.0, 7.0, 7.0, 6.0]
        return _pack(prompt, ans, 30, difficulty, rubric, vocab_size)
    if kind == "signs":
        prompt = "A learner changes +5 to -5 when moving it across the equals sign. What error occurred?"
        ans = "Sign error / inverse operation error."
        rubric = [8.0, 8.0, 6.0, 7.0]
        return _pack(prompt, ans, 31, difficulty, rubric, vocab_size)
    prompt = "A student says the same explanation works for every problem. What coaching response helps?"
    ans = "Use a counterexample and ask them to compare cases."
    rubric = [7.0, 6.0, 8.0, 7.0]
    return _pack(prompt, ans, 32, difficulty, rubric, vocab_size)


def generate_reasoning_example(kind: str = "logic", difficulty: float = 0.5, vocab_size: int = 256) -> Batch:
    if kind == "logic":
        prompt = "All engineers like puzzles. Maya likes puzzles. Can we conclude Maya is an engineer?"
        ans = "No, the premise is not sufficient."
        rubric = [8.0, 6.0, 7.0, 8.0]
        return _pack(prompt, ans, 40, difficulty, rubric, vocab_size)
    if kind == "sequence":
        prompt = "What comes next: 2, 6, 12, 20, ?"
        ans = "30."
        rubric = [6.0, 8.0, 7.0, 6.0]
        return _pack(prompt, ans, 41, difficulty, rubric, vocab_size)
    prompt = "A claim is supported by data in one experiment but not another. What is the cautious conclusion?"
    ans = "The claim is context-dependent and needs more evidence."
    rubric = [7.0, 7.0, 8.0, 7.0]
    return _pack(prompt, ans, 42, difficulty, rubric, vocab_size)
