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
from ..losses import cosine_alignment
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


def id_digest(examples: Sequence[ARCExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.item_id.encode("utf-8"))
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
    tokens = torch.stack(
        [text_to_tokens(format_prompt(ex), vocab_size=vocab_size, max_len=max_len) for ex in examples]
    )
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
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.float().mean().item()) * abs(
                float(correct[mask].mean().item()) - float(confidence[mask].mean().item())
            )
    return {
        "brier": brier,
        "ece": ece,
        "mean_true_class_probability": float(selected.mean().item()),
    }


class HashEncoderClassifier(nn.Module):
    """Shared-input supervised encoder baseline without JEPA/planner/memory/target machinery."""

    def __init__(self, cfg: LAMJEPAConfig, num_choices: int = 4):
        super().__init__()
        self.encoder = MultiViewEncoder(cfg)
        self.projector = nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.classifier = nn.Linear(cfg.proj_dim, num_choices)

    def forward(self, tokens: torch.Tensor, numeric_x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.projector(self.encoder(tokens, numeric_x=numeric_x)))


class LAMARCClassifier(nn.Module):
    """LAM-JEPA backbone with a dedicated four-choice ARC answer head."""

    def __init__(self, cfg: LAMJEPAConfig, num_choices: int = 4):
        super().__init__()
        self.backbone = LAMJEPA(cfg)
        self.choice_head = nn.Linear(cfg.proj_dim, num_choices)

    def forward(
        self,
        tokens: torch.Tensor,
        numeric_x: torch.Tensor,
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
        return self.choice_head(outputs["latent_summary"]), outputs


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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


def _lam_arc_loss(choice_logits: torch.Tensor, outputs: dict, labels: torch.Tensor) -> torch.Tensor:
    supervised = F.cross_entropy(choice_logits, labels)
    align = cosine_alignment(outputs["z_q"], outputs["target_z"])
    quant = outputs["quant_loss"]
    trajectory = outputs["z"].new_tensor(0.0)
    if len(outputs["traj"]) > 1:
        trajectory = torch.stack(
            [F.mse_loss(state, outputs["z_q"].detach()) for state in outputs["traj"][1:]]
        ).mean()
    return supervised + 0.5 * align + 0.25 * quant + 0.25 * trajectory


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
) -> LAMARCClassifier:
    if model_steps < 1:
        raise ValueError("ARC LAM-JEPA benchmark requires model_steps >= 1")
    set_seed(seed)
    model = LAMARCClassifier(cfg, num_choices=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in iter_minibatches(train, batch_size, seed + epoch):
            tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
            optimizer.zero_grad(set_to_none=True)
            choice_logits, outputs = model(
                tokens,
                numeric_x,
                model_steps=model_steps,
                deterministic=False,
            )
            loss = _lam_arc_loss(choice_logits, outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.backbone.update_target()
    return model.eval()


@torch.no_grad()
def _predict_lam(
    model: LAMARCClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
        batch_probs = torch.softmax(logits, dim=-1).cpu()
        probs.append(batch_probs)
        labels.append(y.cpu())
        if len(outputs["actions"]) != model_steps:
            raise RuntimeError(
                f"LAM ARC evaluation expected {model_steps} planner steps, got {len(outputs['actions'])}"
            )
        for example, probability, label in zip(batch, batch_probs, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )
    return torch.cat(probs), torch.cat(labels), rows


@torch.no_grad()
def _predict_baseline(
    model: HashEncoderClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        batch_probs = torch.softmax(model(tokens, numeric_x), dim=-1).cpu()
        probs.append(batch_probs)
        labels.append(y.cpu())
        for example, probability, label in zip(batch, batch_probs, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )
    return torch.cat(probs), torch.cat(labels), rows


def score_predictions(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = probabilities.argmax(dim=1)
    accuracy = float(predictions.eq(labels).float().mean().item())
    return {"accuracy": accuracy, **calibration_metrics(probabilities, labels)}


def majority_reference(train: Sequence[ARCExample], evaluation: Sequence[ARCExample]) -> dict[str, float | int]:
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
    normalized_seeds = [int(seed) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds) or not normalized_seeds:
        raise ValueError("seeds must be non-empty and unique")
    train_data = list(train[:train_limit] if train_limit else train)
    validation_data = list(validation[:validation_limit] if validation_limit else validation)
    if not train_data or not validation_data:
        raise ValueError("ARC train and validation data must be non-empty")
    if any(len(example.choices) != 4 for example in train_data + validation_data):
        raise ValueError("current ARC benchmark protocol requires exactly four choices per example")
    train_ids = {example.item_id for example in train_data}
    validation_ids = {example.item_id for example in validation_data}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise ValueError(f"ARC train/validation ID leakage detected: {overlap[:5]}")

    cfg = LAMJEPAConfig()
    records: list[dict] = []
    reversed_validation = [reverse_choices(example) for example in validation_data]
    expected_ids = [example.item_id for example in validation_data]

    for seed in normalized_seeds:
        baseline = _train_supervised_baseline(
            train_data,
            cfg=cfg,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
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
        baseline_probs, labels, baseline_rows = _predict_baseline(
            baseline, validation_data, cfg=cfg, batch_size=batch_size, device=device
        )
        lam_probs, lam_labels, lam_rows = _predict_lam(
            lam,
            validation_data,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        reversed_probs, reversed_labels, reversed_rows = _predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=batch_size,
            model_steps=model_steps,
            device=device,
        )
        if not torch.equal(labels, lam_labels):
            raise RuntimeError("baseline and LAM-JEPA labels are not aligned")
        if [row["id"] for row in baseline_rows] != expected_ids or [row["id"] for row in lam_rows] != expected_ids:
            raise RuntimeError("validation row identity changed between models")
        if [row["id"] for row in reversed_rows] != expected_ids:
            raise RuntimeError("choice-order robustness rows do not align with canonical validation rows")
        records.append(
            {
                "seed": seed,
                "parameter_counts": {
                    "hash_encoder_baseline": parameter_count(baseline),
                    "lam_jepa_arc": parameter_count(lam),
                },
                "hash_encoder_baseline": score_predictions(baseline_probs, labels),
                "lam_jepa": score_predictions(lam_probs, lam_labels),
                "lam_jepa_reversed_choices": score_predictions(reversed_probs, reversed_labels),
                "raw_predictions": {
                    "hash_encoder_baseline": baseline_rows,
                    "lam_jepa": lam_rows,
                    "lam_jepa_reversed_choices": reversed_rows,
                },
            }
        )

    majority = majority_reference(train_data, validation_data)
    return {
        "protocol": {
            "dataset": "AI2 ARC-Challenge",
            "train_digest": dataset_digest(train_data),
            "validation_digest": dataset_digest(validation_data),
            "train_id_digest": id_digest(train_data),
            "validation_id_digest": id_digest(validation_data),
            "train_validation_overlap": 0,
            "seeds": normalized_seeds,
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
            "baseline_boundary": "The hash-encoder baseline shares the input encoder family but is not parameter matched and is not the required strong pretrained baseline.",
            "claim_boundary": "CI/small-budget runs certify external-data plumbing only. Scientific comparison requires the full preregistered budget, >=5 seeds, a parameter-matched baseline, a strong pretrained baseline, locked test-set evaluation, and independent reproduction."
        },
        "majority_reference": majority,
        "records": records,
    }
