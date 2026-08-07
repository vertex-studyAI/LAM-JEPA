from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    LAMARCClassifier,
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


def resolve_snapshot_revision() -> str:
    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename="config.json",
            revision=MODEL_REVISION,
        )
    ).resolve()
    parts = config_path.parts
    if "snapshots" not in parts:
        raise RuntimeError(f"unexpected Hugging Face cache path: {config_path}")
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        raise RuntimeError(f"snapshot revision missing from cache path: {config_path}")
    return parts[index + 1]


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def tokenize_multiple_choice_batch(
    tokenizer,
    examples: Sequence[ARCExample],
    *,
    max_length: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not examples:
        raise ValueError("cannot tokenize an empty ARC batch")
    if any(len(example.choices) != 4 for example in examples):
        raise ValueError("pretrained ARC baseline requires exactly four choices per item")

    questions: list[str] = []
    choices: list[str] = []
    for example in examples:
        for choice in example.choices:
            questions.append(example.question)
            choices.append(choice)

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


def iter_batches(examples: Sequence[ARCExample], batch_size: int, seed: int) -> list[list[ARCExample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(examples), generator=generator).tolist()
    return [
        [examples[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def load_pretrained_model_and_tokenizer(device: str):
    resolved_revision = resolve_snapshot_revision()
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            f"resolved model revision changed: expected={MODEL_REVISION} actual={resolved_revision}"
        )
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "transformers version mismatch: "
            f"expected={PINNED_TRANSFORMERS_VERSION} actual={transformers.__version__}"
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
    if config_revision is not None and config_revision != MODEL_REVISION:
        raise RuntimeError(
            f"model config resolved unexpected commit: expected={MODEL_REVISION} actual={config_revision}"
        )
    return tokenizer, model, resolved_revision


def train_pretrained_baseline(
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
    tokenizer, model, resolved_revision = load_pretrained_model_and_tokenizer(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_executed = 0

    model.train()
    for epoch in range(epochs):
        for batch in iter_batches(train, batch_size, seed + epoch):
            inputs, labels = tokenize_multiple_choice_batch(
                tokenizer,
                batch,
                max_length=max_length,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("pretrained baseline produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps_executed += 1
            if max_train_steps is not None and steps_executed >= max_train_steps:
                break
        if max_train_steps is not None and steps_executed >= max_train_steps:
            break

    if steps_executed < 1:
        raise RuntimeError("pretrained baseline did not execute a training step")
    return tokenizer, model.eval(), resolved_revision, steps_executed


@torch.no_grad()
def predict_pretrained(
    tokenizer,
    model,
    examples: Sequence[ARCExample],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        inputs, y = tokenize_multiple_choice_batch(
            tokenizer,
            batch,
            max_length=max_length,
            device=device,
        )
        logits = model(**inputs).logits
        batch_probabilities = torch.softmax(logits, dim=-1).cpu()
        probabilities.append(batch_probabilities)
        labels.append(y.cpu())
        for example, probability, label in zip(batch, batch_probabilities, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )

    return torch.cat(probabilities), torch.cat(labels), rows


@torch.no_grad()
def predict_lam(
    model: LAMARCClassifier,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    from lam_jepa.benchmarking.arc_challenge import batchify

    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        tokens, numeric_x, y = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
        if len(outputs["actions"]) != model_steps:
            raise RuntimeError(
                f"LAM-JEPA evaluation expected {model_steps} planner steps, got {len(outputs['actions'])}"
            )
        batch_probabilities = torch.softmax(logits, dim=-1).cpu()
        probabilities.append(batch_probabilities)
        labels.append(y.cpu())
        for example, probability, label in zip(batch, batch_probabilities, y.cpu(), strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )

    return torch.cat(probabilities), torch.cat(labels), rows


def mean_std(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty list")
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LAM-JEPA with an immutable DistilRoBERTa multiple-choice ARC baseline."
    )
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
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8 or args.model_steps < 1:
        parser.error("invalid positive training/evaluation argument")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be at least 1 when supplied")

    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two unique values")

    full_train = load_arc_split(args.train)
    full_validation = load_arc_split(args.validation)
    train = list(full_train[: args.train_limit] if args.train_limit else full_train)
    validation = list(full_validation[: args.validation_limit] if args.validation_limit else full_validation)
    if not train or not validation:
        parser.error("train and validation splits must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        parser.error("the pretrained ARC protocol currently requires exactly four choices")

    overlap = sorted({example.item_id for example in train} & {example.item_id for example in validation})
    if overlap:
        raise SystemExit(f"train/validation leakage detected: {overlap[:5]}")

    cfg = LAMJEPAConfig()
    reversed_validation = [reverse_choices(example) for example in validation]
    records: list[dict[str, object]] = []
    lam_accuracies: list[float] = []
    pretrained_accuracies: list[float] = []
    deltas: list[float] = []
    pretrained_parameter_count: int | None = None
    resolved_revision: str | None = None
    executed_step_counts: list[int] = []

    for seed in seeds:
        tokenizer, pretrained, seed_revision, steps_executed = train_pretrained_baseline(
            train,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_train_steps=args.max_train_steps,
            device=args.device,
        )
        if seed_revision != MODEL_REVISION:
            raise RuntimeError("pretrained model revision changed between resolution and training")
        if resolved_revision is None:
            resolved_revision = seed_revision
        elif resolved_revision != seed_revision:
            raise RuntimeError("pretrained model revision changed across seeds")

        current_parameters = trainable_parameter_count(pretrained)
        if pretrained_parameter_count is None:
            pretrained_parameter_count = current_parameters
        elif pretrained_parameter_count != current_parameters:
            raise RuntimeError("pretrained parameter count changed across seeds")
        executed_step_counts.append(steps_executed)

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
            tokenizer,
            pretrained,
            validation,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        lam_probs, lam_labels, lam_rows = predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        if not torch.equal(pretrained_labels, lam_labels):
            raise RuntimeError("pretrained baseline and LAM-JEPA evaluated different validation labels")

        pretrained_reverse_probs, pretrained_reverse_labels, pretrained_reverse_rows = predict_pretrained(
            tokenizer,
            pretrained,
            reversed_validation,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        lam_reverse_probs, lam_reverse_labels, lam_reverse_rows = predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        if not torch.equal(pretrained_reverse_labels, lam_reverse_labels):
            raise RuntimeError("pretrained baseline and LAM-JEPA evaluated different reversed labels")

        pretrained_metrics = score_predictions(pretrained_probs, pretrained_labels)
        lam_metrics = score_predictions(lam_probs, lam_labels)
        pretrained_reverse_metrics = score_predictions(pretrained_reverse_probs, pretrained_reverse_labels)
        lam_reverse_metrics = score_predictions(lam_reverse_probs, lam_reverse_labels)
        delta = float(lam_metrics["accuracy"] - pretrained_metrics["accuracy"])
        pretrained_accuracies.append(float(pretrained_metrics["accuracy"]))
        lam_accuracies.append(float(lam_metrics["accuracy"]))
        deltas.append(delta)

        records.append(
            {
                "seed": seed,
                "pretrained_baseline": {
                    "metrics": pretrained_metrics,
                    "choice_reversal_metrics": pretrained_reverse_metrics,
                    "predictions": pretrained_rows,
                    "choice_reversal_predictions": pretrained_reverse_rows,
                    "training_steps_executed": steps_executed,
                },
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_reverse_metrics,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_reverse_rows,
                },
                "accuracy_delta_lam_minus_pretrained": delta,
            }
        )

        del pretrained
        del tokenizer

    assert pretrained_parameter_count is not None
    assert resolved_revision is not None
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
            "pretrained_model_trainable_parameters": pretrained_parameter_count,
            "transformers_version": transformers.__version__,
            "transformers_version_pin": PINNED_TRANSFORMERS_VERSION,
            "pretrained_training_steps_by_seed": executed_step_counts,
            "comparison_type": "strong pretrained language-model baseline; capacity and compute are not matched to LAM-JEPA",
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
            "lam_accuracy": mean_std(lam_accuracies),
            "pretrained_accuracy": mean_std(pretrained_accuracies),
            "paired_accuracy_delta_lam_minus_pretrained": mean_std(deltas),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
