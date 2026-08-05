from __future__ import annotations

from hashlib import blake2b
from typing import Callable, Sequence

import torch

from ..data import Batch, sample_batch
from ..edtech.tasks import (
    generate_math_example,
    generate_reading_example,
    generate_reasoning_example,
    generate_tutoring_example,
)


TARGET_SEMANTICS = {
    "parity": "answer_class",
    "modadd": "answer_class",
    "algebra": "answer_class",
    "chain": "answer_class",
    "gsm8k": "concept_proxy",
    "equation": "answer_class",
    "science": "answer_class",
    "reading": "concept_proxy",
    "tutoring": "concept_proxy",
    "reasoning": "concept_proxy",
}

_GENERATED_TASKS: dict[str, tuple[Callable[..., Batch], tuple[str, ...]]] = {
    "gsm8k": (generate_math_example, ("gsm8k",)),
    "reading": (generate_reading_example, ("main_idea", "evidence", "comprehension")),
    "tutoring": (generate_tutoring_example, ("fractions", "signs", "misconception")),
    "reasoning": (generate_reasoning_example, ("logic", "sequence", "cautious_conclusion")),
}


def _row_fingerprint(tokens: torch.Tensor, numeric_x: torch.Tensor | None) -> str:
    digest = blake2b(digest_size=16)
    digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
    if numeric_x is not None:
        digest.update(numeric_x.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _annotate_batch(batch: Batch, sampling_unit: str) -> Batch:
    fingerprints = [
        _row_fingerprint(
            batch.tokens[index],
            batch.numeric_x[index] if batch.numeric_x is not None else None,
        )
        for index in range(batch.labels.shape[0])
    ]
    metadata = dict(batch.metadata)
    metadata.update(
        {
            "sampling_unit": sampling_unit,
            "input_fingerprints": fingerprints,
            "unique_inputs": len(set(fingerprints)),
        }
    )
    batch.metadata = metadata
    return batch


def _stack_examples(examples: Sequence[Batch], task: str) -> Batch:
    if not examples:
        raise ValueError("at least one generated example is required")

    prompts = [example.prompt for example in examples if example.prompt]
    solutions = [example.solution for example in examples if example.solution]
    concepts = [example.concept for example in examples if example.concept]
    difficulties = [float(example.difficulty) for example in examples if example.difficulty is not None]

    batch = Batch(
        tokens=torch.cat([example.tokens for example in examples], dim=0),
        numeric_x=torch.cat([example.numeric_x for example in examples], dim=0),
        labels=torch.cat([example.labels for example in examples], dim=0),
        rubric=torch.cat([example.rubric for example in examples], dim=0),
        prompt=examples[0].prompt if len(examples) == 1 else None,
        solution=examples[0].solution if len(examples) == 1 else None,
        concept=examples[0].concept if len(set(concepts)) == 1 and concepts else f"mixed_{task}",
        difficulty=sum(difficulties) / len(difficulties) if difficulties else None,
        metadata={
            "task": task,
            "prompts": prompts,
            "solutions": solutions,
            "concepts": concepts,
            "unique_prompts": len(set(prompts)),
            "generated_examples": len(examples),
        },
    )
    return _annotate_batch(batch, sampling_unit="generated_example")


def _sample_generated_batch(task: str, batch_size: int, vocab_size: int) -> Batch:
    generator, kinds = _GENERATED_TASKS[task]
    offset = int(torch.randint(0, len(kinds), (1,)).item()) if len(kinds) > 1 else 0
    examples: list[Batch] = []
    seen_prompts: set[str] = set()

    for index in range(batch_size):
        kind = kinds[(offset + index) % len(kinds)]
        candidate = None
        for _ in range(32):
            difficulty = 0.2 + 0.6 * float(torch.rand(1).item())
            candidate = generator(kind, difficulty=difficulty, vocab_size=vocab_size)
            if task != "gsm8k" or candidate.prompt not in seen_prompts:
                break
        if candidate is None:
            raise RuntimeError(f"failed to generate evaluation example for {task}")
        examples.append(candidate)
        if candidate.prompt:
            seen_prompts.add(candidate.prompt)

    return _stack_examples(examples, task)


def sample_evaluation_batch(task: str, batch_size: int, vocab_size: int) -> Batch:
    """Sample evaluation rows without repeating one generated example across a batch.

    The regular procedural tasks already produce one independent row per requested
    batch element. Generated text-style tasks are sampled one example at a time and
    stacked so that ``n`` is not silently inflated by tensor repetition.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if task not in TARGET_SEMANTICS:
        raise ValueError(f"unknown evaluation task: {task}")

    if task in _GENERATED_TASKS:
        return _sample_generated_batch(task, batch_size, vocab_size)

    return _annotate_batch(
        sample_batch(task, batch=batch_size, vocab_size=vocab_size),
        sampling_unit="procedural_row",
    )
