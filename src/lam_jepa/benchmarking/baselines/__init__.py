from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

import torch

from ..edtech_suite import EDTECH_TASKS
from ..evaluation_sampling import TARGET_SEMANTICS, sample_evaluation_batch


@torch.no_grad()
def evaluate_label_baselines(
    tasks: Sequence[str] = EDTECH_TASKS,
    *,
    batch_size: int = 64,
    batches: int = 8,
    vocab_size: int = 256,
) -> Dict[str, Dict[str, float | int | str]]:
    """Measure label-distribution references with the benchmark sampler.

    No model is trained or executed. The majority result is an oracle
    class-frequency reference over sampled evaluation labels, not a deployable
    predictor. The two uniform results are expected guessing accuracies over
    the sampled label support and configured output vocabulary.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if batches < 1:
        raise ValueError("batches must be at least 1")
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")

    out: Dict[str, Dict[str, float | int | str]] = {}
    for task in tasks:
        labels: list[int] = []
        input_fingerprints: set[str] = set()
        prompts: set[str] = set()

        for _ in range(batches):
            batch = sample_evaluation_batch(task, batch_size=batch_size, vocab_size=vocab_size)
            labels.extend(int(value) for value in batch.labels.detach().cpu().reshape(-1).tolist())

            fingerprints = batch.metadata.get("input_fingerprints", [])
            if isinstance(fingerprints, list):
                input_fingerprints.update(str(value) for value in fingerprints)
            batch_prompts = batch.metadata.get("prompts", [])
            if isinstance(batch_prompts, list):
                prompts.update(str(value) for value in batch_prompts if value)

        if not labels:
            raise ValueError(f"task {task!r} produced no labels")
        if min(labels) < 0 or max(labels) >= vocab_size:
            raise ValueError(
                f"task {task!r} produced labels outside [0, {vocab_size}): "
                f"min={min(labels)}, max={max(labels)}"
            )

        counts = Counter(labels)
        majority_count = max(counts.values())
        majority_label = min(label for label, count in counts.items() if count == majority_count)
        unique_labels = len(counts)
        n = len(labels)

        out[task] = {
            "n": n,
            "unique_inputs": len(input_fingerprints),
            "unique_labels": unique_labels,
            "unique_prompts": len(prompts),
            "majority_label": majority_label,
            "majority_count": majority_count,
            "majority_accuracy": majority_count / n,
            "uniform_observed_label_accuracy": 1.0 / unique_labels,
            "uniform_full_vocab_accuracy": 1.0 / vocab_size,
            "vocab_size": vocab_size,
            "target_semantics": TARGET_SEMANTICS[task],
            "baseline_semantics": "sampled-label-distribution reference; no model executed",
        }

    return out


__all__ = ["evaluate_label_baselines"]
