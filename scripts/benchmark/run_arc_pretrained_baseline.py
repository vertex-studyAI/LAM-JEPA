from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import transformers
from huggingface_hub import HfApi
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    _predict_lam,
    _train_lam_jepa,
    dataset_digest,
    id_digest,
    load_arc_split,
    reverse_choices,
    score_predictions,
)
from lam_jepa.model import LAMJEPAConfig
from lam_jepa.utils import set_seed

MODEL_ID = "distilbert/distilroberta-base"
MODEL_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"
MODEL_LICENSE = "Apache-2.0"
PINNED_TRANSFORMERS_VERSION = "4.57.6"


def verify_remote_revision() -> str:
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
    resolved = str(info.sha)
    if resolved != MODEL_REVISION:
        raise RuntimeError(f"pretrained revision mismatch: expected={MODEL_REVISION} actual={resolved}")
    return resolved


def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def encode_batch(tokenizer, examples: Sequence[ARCExample], max_length: int, device: str):
    if not examples or any(len(example.choices) != 4 for example in examples):
        raise ValueError("pretrained ARC batches must be non-empty with exactly four choices")
    questions = [example.question for example in examples for _ in example.choices]
    choices = [choice for example in examples for choice in example.choices]
    encoded = tokenizer(
        questions,
        choices,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_size = len(examples)
    inputs = {
        key: value.reshape(batch_size, 4, -1).to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    labels = torch.tensor([example.label for example in examples], dtype=torch.long, device=device)
    return inputs, labels


def iter_batches(examples: Sequence[ARCExample], batch_size: int, seed: int):
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def load_pretrained(device: str):
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version mismatch: expected={PINNED_TRANSFORMERS_VERSION} actual={transformers.__version__}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    config_revision = getattr(model.config, "_commit_hash", None)
    if config_revision is not None and str(config_revision) != MODEL_REVISION:
        raise RuntimeError(f"loaded config revision mismatch: {config_revision}")
    return tokenizer, model


def train_pretrained(
    train: Sequence[ARCExample],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    max_train_steps: int | None,
    device: str,
):
    set_seed(seed)
    transformers.set_seed(seed)
    tokenizer, model = load_pretrained(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = 0
    model.train()
    for epoch in range(epochs):
        for batch in iter_batches(train, batch_size, seed + epoch):
            inputs, labels = encode_batch(tokenizer, batch, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("pretrained baseline produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if max_train_steps is not None and steps >= max_train_steps:
                break
        if max_train_steps is not None and steps >= max_train_steps:
            break
    if steps < 1:
        raise RuntimeError("pretrained baseline executed zero training steps")
    return tokenizer, model.eval(), steps


@torch.no_grad()
def predict_pretrained(tokenizer, model, examples, *, batch_size: int, max_length: int, device: str):
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        inputs, labels = encode_batch(tokenizer, batch, max_length, device)
        probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
        all_probabilities.append(probabilities)
        all_labels.append(labels.cpu())
        for example, probability, label in zip(batch, probabilities, labels.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )
    return torch.cat(all_probabilities), torch.cat(all_labels), rows


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an immutable DistilRoBERTa ARC comparison.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.model_steps) < 1 or args.max_length < 8:
        parser.error("epochs, batch size, model steps, and max length must be positive")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be at least 1")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two unique values")

    train_all = load_arc_split(args.train)
    validation_all = load_arc_split(args.validation)
    train = list(train_all[: args.train_limit] if args.train_limit else train_all)
    validation = list(validation_all[: args.validation_limit] if args.validation_limit else validation_all)
    if not train or not validation:
        parser.error("train and validation splits must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        parser.error("current pretrained ARC protocol requires exactly four choices")
    overlap = {example.item_id for example in train} & {example.item_id for example in validation}
    if overlap:
        raise SystemExit(f"ARC train/validation leakage detected: {sorted(overlap)[:5]}")

    resolved_revision = verify_remote_revision()
    cfg = LAMJEPAConfig()
    reversed_validation = [reverse_choices(example) for example in validation]
    records: list[dict[str, object]] = []
    lam_accuracies: list[float] = []
    pretrained_accuracies: list[float] = []
    deltas: list[float] = []
    parameter_count: int | None = None
    training_steps: list[int] = []

    for seed in seeds:
        tokenizer, pretrained, steps = train_pretrained(
            train,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_train_steps=args.max_train_steps,
            device=args.device,
        )
        current_count = trainable_parameters(pretrained)
        if parameter_count is None:
            parameter_count = current_count
        elif current_count != parameter_count:
            raise RuntimeError("pretrained parameter count changed across seeds")
        training_steps.append(steps)

        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_steps=args.model_steps,
            device=args.device,
        )
        pretrained_probs, pretrained_labels, pretrained_rows = predict_pretrained(
            tokenizer, pretrained, validation,
            batch_size=args.batch_size, max_length=args.max_length, device=args.device,
        )
        lam_probs, lam_labels, lam_rows = _predict_lam(
            lam, validation, cfg=cfg, batch_size=args.batch_size, model_steps=args.model_steps, device=args.device,
        )
        if not torch.equal(pretrained_labels, lam_labels):
            raise RuntimeError("LAM-JEPA and pretrained baseline evaluated different validation labels")

        pretrained_rev_probs, pretrained_rev_labels, pretrained_rev_rows = predict_pretrained(
            tokenizer, pretrained, reversed_validation,
            batch_size=args.batch_size, max_length=args.max_length, device=args.device,
        )
        lam_rev_probs, lam_rev_labels, lam_rev_rows = _predict_lam(
            lam, reversed_validation, cfg=cfg, batch_size=args.batch_size, model_steps=args.model_steps, device=args.device,
        )
        if not torch.equal(pretrained_rev_labels, lam_rev_labels):
            raise RuntimeError("LAM-JEPA and pretrained baseline evaluated different reversed labels")

        pretrained_metrics = score_predictions(pretrained_probs, pretrained_labels)
        lam_metrics = score_predictions(lam_probs, lam_labels)
        pretrained_rev_metrics = score_predictions(pretrained_rev_probs, pretrained_rev_labels)
        lam_rev_metrics = score_predictions(lam_rev_probs, lam_rev_labels)
        delta = float(lam_metrics["accuracy"] - pretrained_metrics["accuracy"])
        pretrained_accuracies.append(float(pretrained_metrics["accuracy"]))
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        deltas.append(delta)
        records.append(
            {
                "seed": seed,
                "pretrained_baseline": {
                    "metrics": pretrained_metrics,
                    "choice_reversal_metrics": pretrained_rev_metrics,
                    "predictions": pretrained_rows,
                    "choice_reversal_predictions": pretrained_rev_rows,
                    "training_steps_executed": steps,
                },
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_rev_metrics,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_rev_rows,
                },
                "accuracy_delta_lam_minus_pretrained": delta,
            }
        )
        del pretrained, tokenizer, lam

    assert parameter_count is not None
    payload = {
        "protocol": {
            "dataset": "AI2 ARC-Challenge",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "max_train_steps": args.max_train_steps,
            "model_steps": args.model_steps,
            "device": args.device,
            "primary_metric": "multiple-choice accuracy",
            "calibration_metrics": ["brier", "ece", "mean_true_class_probability"],
            "robustness_check": "deterministic reversal of answer-choice order with label remapping",
            "pretrained_model_id": MODEL_ID,
            "pretrained_model_revision": MODEL_REVISION,
            "resolved_pretrained_revision": resolved_revision,
            "pretrained_model_license": MODEL_LICENSE,
            "pretrained_model_trainable_parameters": parameter_count,
            "transformers_version": transformers.__version__,
            "transformers_version_pin": PINNED_TRANSFORMERS_VERSION,
            "pretrained_training_steps_by_seed": training_steps,
            "comparison_type": "pretrained language-model baseline; capacity and compute are not matched to LAM-JEPA",
            "final_seed_requirement": 5,
            "test_split_policy": "not downloaded or evaluated by this development command",
            "claim_boundary": (
                "This bounded development smoke verifies immutable pretrained-baseline execution and paired ARC evaluation only. "
                "It is not the final >=5-seed budget, not a locked-test result, not compute-matched, not independent "
                "reproduction, and not evidence that either model is superior or RESEARCH_COMPLETE."
            ),
        },
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_accuracies),
            "pretrained_accuracy": summarize(pretrained_accuracies),
            "paired_accuracy_delta_lam_minus_pretrained": summarize(deltas),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
