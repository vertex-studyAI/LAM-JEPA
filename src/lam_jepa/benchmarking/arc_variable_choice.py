from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import text_to_tokens
from ..model import LAMJEPA, LAMJEPAConfig, MultiViewEncoder
from ..utils import set_seed
from .arc_challenge import ARCExample, _lam_arc_loss, dataset_digest, id_digest


@dataclass(frozen=True)
class VariableChoiceBatch:
    tokens: torch.Tensor
    numeric_x: torch.Tensor
    choice_counts: tuple[int, ...]
    labels: torch.Tensor


def format_candidate(example: ARCExample, choice: str) -> str:
    return f"Question: {example.question} Candidate answer: {choice}"


def batchify_variable(
    examples: Sequence[ARCExample],
    *,
    vocab_size: int,
    numeric_dim: int = 4,
    max_len: int = 96,
    device: torch.device | str = "cpu",
) -> VariableChoiceBatch:
    if not examples:
        raise ValueError("cannot batch an empty ARC example list")
    if any(len(example.choices) < 2 for example in examples):
        raise ValueError("every ARC item must have at least two candidate choices")
    if numeric_dim < 1:
        raise ValueError("numeric_dim must be positive")

    candidate_tokens = [
        text_to_tokens(format_candidate(example, choice), vocab_size=vocab_size, max_len=max_len)
        for example in examples
        for choice in example.choices
    ]
    tokens = torch.stack(candidate_tokens)
    numeric_x = torch.zeros(tokens.size(0), numeric_dim, dtype=torch.float32)
    labels = torch.tensor([example.label for example in examples], dtype=torch.long)
    counts = tuple(len(example.choices) for example in examples)
    for example, count in zip(examples, counts, strict=True):
        if not 0 <= example.label < count:
            raise ValueError(f"{example.item_id}: answer index {example.label} invalid for {count} choices")

    target = torch.device(device)
    return VariableChoiceBatch(
        tokens=tokens.to(target),
        numeric_x=numeric_x.to(target),
        choice_counts=counts,
        labels=labels.to(target),
    )


def pack_candidate_scores(flat_scores: torch.Tensor, choice_counts: Sequence[int]) -> torch.Tensor:
    if flat_scores.dim() != 1:
        raise ValueError("flat candidate scores must be one-dimensional")
    if not choice_counts or any(count < 2 for count in choice_counts):
        raise ValueError("choice counts must contain values >= 2")
    if sum(choice_counts) != flat_scores.numel():
        raise ValueError("candidate-score count does not match question boundaries")

    max_choices = max(choice_counts)
    padded = flat_scores.new_full((len(choice_counts), max_choices), -1e9)
    offset = 0
    for row, count in enumerate(choice_counts):
        padded[row, :count] = flat_scores[offset : offset + count]
        offset += count
    return padded


class VariableChoiceLAMClassifier(nn.Module):
    def __init__(self, cfg: LAMJEPAConfig):
        super().__init__()
        self.backbone = LAMJEPA(cfg)
        self.answer_scorer = nn.Linear(cfg.proj_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        numeric_x: torch.Tensor,
        choice_counts: Sequence[int],
        *,
        model_steps: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, dict]:
        outputs = self.backbone(
            tokens,
            numeric_x=numeric_x,
            steps=model_steps,
            sample_rollout=not deterministic,
            noise_std=0.0 if deterministic else None,
        )
        flat_scores = self.answer_scorer(outputs["latent_summary"]).squeeze(-1)
        return pack_candidate_scores(flat_scores, choice_counts), outputs


class VariableChoiceHashEncoder(nn.Module):
    """Non-JEPA shared candidate scorer for protocol-v3 execution plumbing."""

    def __init__(self, cfg: LAMJEPAConfig):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.answer_scorer = nn.Linear(cfg.proj_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        numeric_x: torch.Tensor,
        choice_counts: Sequence[int],
    ) -> torch.Tensor:
        representation = self.projector(self.encoder(tokens, numeric_x=numeric_x))
        flat_scores = self.answer_scorer(representation).squeeze(-1)
        return pack_candidate_scores(flat_scores, choice_counts)


def iter_minibatches(examples: Sequence[ARCExample], batch_size: int, seed: int) -> Iterable[list[ARCExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def train_variable_lam(
    train: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    model_steps: int,
    device: str,
) -> VariableChoiceLAMClassifier:
    set_seed(seed)
    model = VariableChoiceLAMClassifier(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for examples in iter_minibatches(train, batch_size, seed + epoch):
            batch = batchify_variable(
                examples,
                vocab_size=cfg.vocab_size,
                numeric_dim=cfg.numeric_dim,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, outputs = model(
                batch.tokens,
                batch.numeric_x,
                batch.choice_counts,
                model_steps=model_steps,
                deterministic=False,
            )
            loss = _lam_arc_loss(logits, outputs, batch.labels)
            if not torch.isfinite(loss):
                raise RuntimeError("variable-choice LAM-JEPA produced a non-finite loss")
            loss.backward()
            optimizer.step()
            model.backbone.update_target()
    return model.eval()


def train_variable_hash(
    train: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> VariableChoiceHashEncoder:
    set_seed(seed)
    model = VariableChoiceHashEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for examples in iter_minibatches(train, batch_size, seed + epoch):
            batch = batchify_variable(
                examples,
                vocab_size=cfg.vocab_size,
                numeric_dim=cfg.numeric_dim,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.tokens, batch.numeric_x, batch.choice_counts)
            loss = F.cross_entropy(logits, batch.labels)
            if not torch.isfinite(loss):
                raise RuntimeError("variable-choice supervised baseline produced a non-finite loss")
            loss.backward()
            optimizer.step()
    return model.eval()


def _row_metrics(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot score empty predictions")
    correct: list[float] = []
    confidences: list[float] = []
    true_probabilities: list[float] = []
    brier_rows: list[float] = []
    for row in rows:
        label = int(row["label"])
        prediction = int(row["prediction"])
        probabilities = [float(value) for value in row["probabilities"]]
        if not 0 <= label < len(probabilities):
            raise ValueError("label outside per-row probability vector")
        if not 0 <= prediction < len(probabilities):
            raise ValueError("prediction outside per-row probability vector")
        correct.append(float(prediction == label))
        confidences.append(max(probabilities))
        true_probabilities.append(probabilities[label])
        brier_rows.append(
            sum(
                (probability - (1.0 if index == label else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )

    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            index
            for index, confidence in enumerate(confidences)
            if (confidence >= lower if bin_index == 0 else confidence > lower) and confidence <= upper
        ]
        if members:
            mean_correct = sum(correct[index] for index in members) / len(members)
            mean_confidence = sum(confidences[index] for index in members) / len(members)
            ece += (len(members) / len(rows)) * abs(mean_correct - mean_confidence)
    return {
        "accuracy": sum(correct) / len(correct),
        "brier": sum(brier_rows) / len(brier_rows),
        "ece": ece,
        "mean_true_class_probability": sum(true_probabilities) / len(true_probabilities),
    }


@torch.no_grad()
def predict_variable_lam(
    model: VariableChoiceLAMClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        batch = batchify_variable(
            current,
            vocab_size=cfg.vocab_size,
            numeric_dim=cfg.numeric_dim,
            device=device,
        )
        logits, outputs = model(
            batch.tokens,
            batch.numeric_x,
            batch.choice_counts,
            model_steps=model_steps,
            deterministic=True,
        )
        if len(outputs["actions"]) != model_steps:
            raise RuntimeError("variable-choice LAM-JEPA did not execute the declared planner steps")
        probabilities = torch.softmax(logits, dim=-1).cpu()
        for example, count, probability in zip(current, batch.choice_counts, probabilities, strict=True):
            actual = probability[:count]
            actual = actual / actual.sum()
            rows.append(
                {
                    "id": example.item_id,
                    "choice_count": count,
                    "label": example.label,
                    "prediction": int(actual.argmax().item()),
                    "probabilities": [float(value) for value in actual.tolist()],
                }
            )
    return _row_metrics(rows), rows


@torch.no_grad()
def predict_variable_hash(
    model: VariableChoiceHashEncoder,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    device: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        batch = batchify_variable(
            current,
            vocab_size=cfg.vocab_size,
            numeric_dim=cfg.numeric_dim,
            device=device,
        )
        logits = model(batch.tokens, batch.numeric_x, batch.choice_counts)
        probabilities = torch.softmax(logits, dim=-1).cpu()
        for example, count, probability in zip(current, batch.choice_counts, probabilities, strict=True):
            actual = probability[:count]
            actual = actual / actual.sum()
            rows.append(
                {
                    "id": example.item_id,
                    "choice_count": count,
                    "label": example.label,
                    "prediction": int(actual.argmax().item()),
                    "probabilities": [float(value) for value in actual.tolist()],
                }
            )
    return _row_metrics(rows), rows


def cardinality_majority_indices(train: Sequence[ARCExample]) -> dict[int, int]:
    by_count: dict[int, Counter] = {}
    for example in train:
        by_count.setdefault(len(example.choices), Counter())[example.label] += 1
    return {
        count: min(
            (label for label, frequency in labels.items() if frequency == max(labels.values())),
            default=0,
        )
        for count, labels in by_count.items()
    }


def predict_cardinality_majority(
    train: Sequence[ARCExample], validation: Sequence[ARCExample]
) -> tuple[dict[str, float], list[dict[str, object]], dict[int, int], list[int]]:
    majority = cardinality_majority_indices(train)
    unseen: list[int] = []
    rows: list[dict[str, object]] = []
    for example in validation:
        count = len(example.choices)
        if count in majority:
            prediction = majority[count]
        else:
            prediction = 0
            unseen.append(count)
        probabilities = [0.0] * count
        probabilities[prediction] = 1.0
        rows.append(
            {
                "id": example.item_id,
                "choice_count": count,
                "label": example.label,
                "prediction": prediction,
                "probabilities": probabilities,
            }
        )
    return _row_metrics(rows), rows, majority, sorted(set(unseen))


def choice_count_distribution(examples: Sequence[ARCExample]) -> dict[str, int]:
    return {str(count): frequency for count, frequency in sorted(Counter(len(example.choices) for example in examples).items())}


def variable_protocol_identity(train: Sequence[ARCExample], validation: Sequence[ARCExample]) -> dict[str, object]:
    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    if overlap:
        raise ValueError(f"train/validation overlap: {overlap[:5]}")
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_digest": dataset_digest(train),
        "validation_digest": dataset_digest(validation),
        "train_id_digest": id_digest(train),
        "validation_id_digest": id_digest(validation),
        "train_validation_overlap": 0,
        "train_choice_count_distribution": choice_count_distribution(train),
        "validation_choice_count_distribution": choice_count_distribution(validation),
    }
