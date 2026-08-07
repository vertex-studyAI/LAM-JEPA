from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import text_to_tokens
from ..losses import total_loss
from ..model import LAMJEPA, LAMJEPAConfig, MultiViewEncoder
from ..utils import set_seed


@dataclass(frozen=True)
class ARCExample:
    item_id: str
    question: str
    choices: tuple[str, ...]
    label: int


def _as_list(value) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise ValueError(f"expected sequence, got {type(value).__name__}")


def load_arc_split(path: str | Path) -> list[ARCExample]:
    frame = pd.read_parquet(path)
    required = {"id", "question", "choices", "answerKey"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ARC parquet missing columns: {sorted(missing)}")

    examples: list[ARCExample] = []
    seen_ids: set[str] = set()
    for row in frame.to_dict(orient="records"):
        item_id = str(row["id"])
        if item_id in seen_ids:
            raise ValueError(f"duplicate ARC id: {item_id}")
        seen_ids.add(item_id)
        choice_obj = row["choices"]
        if not isinstance(choice_obj, dict):
            raise ValueError(f"{item_id}: choices must be a mapping")
        texts = [str(value) for value in _as_list(choice_obj["text"])]
        labels = [str(value) for value in _as_list(choice_obj["label"])]
        if len(texts) != len(labels) or len(texts) < 2:
            raise ValueError(f"{item_id}: invalid choice structure")
        answer = str(row["answerKey"])
        if answer not in labels:
            raise ValueError(f"{item_id}: answer key {answer!r} not present in choice labels {labels}")
        examples.append(
            ARCExample(
                item_id=item_id,
                question=str(row["question"]),
                choices=tuple(texts),
                label=labels.index(answer),
            )
        )
    return examples


def dataset_digest(examples: Sequence[ARCExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        payload = json.dumps(
            {
                "id": example.item_id,
                "question": example.question,
                "choices": example.choices,
                "label": example.label,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def format_prompt(example: ARCExample) -> str:
    options = " ".join(f"[{index}] {text}" for index, text in enumerate(example.choices))
    return f"Question: {example.question} Choices: {options}"


def reverse_choices(example: ARCExample) -> ARCExample:
    choices = tuple(reversed(example.choices))
    return ARCExample(
        item_id=example.item_id,
        question=example.question,
        choices=choices,
        label=len(example.choices) - 1 - example.label,
    )


def batchify(
    examples: Sequence[ARCExample],
    *,
    vocab_size: int,
    max_len: int = 96,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot batch an empty ARC example list")
    tokens = torch.stack([text_to_tokens(format_prompt(ex), vocab_size=vocab_size, max_len=max_len) for ex in examples])
    numeric_x = torch.zeros(len(examples), 1, dtype=torch.float32)
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.long)
    device = torch.device(device)
    return tokens.to(device), numeric_x.to(device), labels.to(device)


def iter_minibatches(examples: Sequence[ARCExample], batch_size: int, seed: int) -> Iterable[list[ARCExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def calibration_metrics(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict[str, float]:
    probabilities = probabilities.detach().cpu().float()
    labels = labels.detach().cpu().long()
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    one_hot = F.one_hot(labels, num_classes=probabilities.shape[1]).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item())

    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper if index else confidence >= lower)
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].mean().item()) - float(confidence[mask].mean().item())
            )
    return {"brier": brier, "ece": ece, "mean_true_class_probability": float(selected.mean().item())}


class HashEncoderClassifier(nn.Module):
    """Matched-input supervised baseline without JEPA/planner/memory/target machinery."""

    def __init__(self, cfg: LAMJEPAConfig, num_choices: int = 4):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.classifier = nn.Linear(cfg.proj_dim, num_choices)

    def forward(self, tokens: torch.Tensor, numeric_x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projector(self.encoder(tokens, numeric_x=numeric_x)))


def _train_supervised_baseline(
    train: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> HashEncoderClassifier:
    set_seed(seed)
    model = HashEncoderClassifier(cfg, num_choices=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tokens, numeric_x), labels)
            loss.backward()
            optimizer.step()
    return model.eval()


def _train_lam_jepa(
    train: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    model_steps: int,
    device: str,
) -> LAMJEPA:
    if model_steps < 1:
        raise ValueError("ARC LAM-JEPA benchmark requires model_steps >= 1")
    set_seed(seed)
    model = LAMJEPA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens, numeric_x=numeric_x, steps=model_steps)
            loss, _ = total_loss(outputs, labels, rubric_target=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
    return model.eval()


@torch.no_grad()
def _predict_lam(
    model: LAMJEPA,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        outputs = model(tokens, numeric_x=numeric_x, steps=model_steps, sample_rollout=False, noise_std=0.0)
        probs.append(torch.softmax(outputs["logits"][:, :4], dim=-1).cpu())
        labels.append(y.cpu())
    return torch.cat(probs), torch.cat(labels)


@torch.no_grad()
def _predict_baseline(
    model: HashEncoderClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        probs.append(torch.softmax(model(tokens, numeric_x), dim=-1).cpu())
        labels.append(y.cpu())
    return torch.cat(probs), torch.cat(labels)


def score_predictions(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = probabilities.argmax(dim=1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    return {"accuracy": accuracy, **calibration_metrics(probabilities, labels)}


def majority_reference(train: Sequence[ARCExample], evaluation: Sequence[ARCExample]) -> dict[str, float]:
    counts = np.bincount([example.label for example in train], minlength=4)
    majority = int(counts.argmax())
    labels = torch.tensor([example.label for example in evaluation])
    probabilities = torch.zeros(len(evaluation), 4)
    probabilities[:, majority] = 1.0
    return {"majority_label": majority, **score_predictions(probabilities, labels)}


def run_arc_smoke(
    train: Sequence[ARCExample],
    validation: Sequence[ARCExample],
    *,
    seeds: Sequence[int] = (1, 2),
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-4,
    model_steps: int = 1,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    device: str = "cpu",
) -> dict:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be non-empty and unique")
    train_data = list(train[:train_limit] if train_limit else train)
    validation_data = list(validation[:validation_limit] if validation_limit else validation)
    if not train_data or not validation_data:
        raise ValueError("ARC train and validation data must be non-empty")
    if any(len(example.choices) != 4 for example in train_data + validation_data):
        raise ValueError("current ARC benchmark protocol requires exactly four choices per example")

    cfg = LAMJEPAConfig()
    records: list[dict] = []
    reversed_validation = [reverse_choices(example) for example in validation_data]

    for seed in [int(value) for value in seeds]:
        baseline = _train_supervised_baseline(
            train_data, cfg=cfg, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, device=device
        )
        lam = _train_lam_jepa(
            train_data,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_steps=model_steps,
            device=device,
        )
        baseline_probs, labels = _predict_baseline(
            baseline, validation_data, cfg=cfg, batch_size=batch_size, device=device
        )
        lam_probs, lam_labels = _predict_lam(
            lam,
            validation_data,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        reversed_probs, reversed_labels = _predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        if not torch.equal(labels, lam_labels):
            raise RuntimeError("baseline and LAM-JEPA labels are not aligned")
        records.append(
            {
                "seed": seed,
                "hash_encoder_baseline": score_predictions(baseline_probs, labels),
                "lam_jepa": score_predictions(lam_probs, lam_labels),
                "lam_jepa_reversed_choices": score_predictions(reversed_probs, reversed_labels),
            }
        )

    majority = majority_reference(train_data, validation_data)
    return {
        "protocol": {
            "dataset": "AI2 ARC-Challenge",
            "train_digest": dataset_digest(train_data),
            "validation_digest": dataset_digest(validation_data),
            "seeds": [int(value) for value in seeds],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_steps": model_steps,
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "primary_metric": "multiple-choice accuracy",
            "calibration_metrics": ["Brier score", "ECE", "mean true-class probability"],
            "robustness_check": "deterministic reversal of answer-choice order with label remapping",
            "test_split_policy": "held out from this command",
            "claim_boundary": "CI/small-budget runs certify plumbing only. Scientific comparison requires the full preregistered budget, >=5 seeds, strong pretrained baseline, test-set lock, and independent reproduction."
        },
        "majority_reference": majority,
        "records": records,
    }
