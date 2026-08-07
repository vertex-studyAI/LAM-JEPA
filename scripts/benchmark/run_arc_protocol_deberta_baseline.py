from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sentencepiece
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

PROTOCOL_PATH = Path("protocols/arc_challenge_v1.json")
EXPECTED_PROTOCOL_ID = "lam-jepa-arc-challenge-v1"
EXPECTED_MODEL_ID = "microsoft/deberta-v3-xsmall"
EXPECTED_MODEL_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
EXPECTED_LICENSE = "MIT"
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_SENTENCEPIECE_VERSION = "0.2.2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_protocol(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen protocol not found: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise RuntimeError("unexpected ARC protocol id")
    if protocol.get("status") != "FROZEN_BEFORE_CONFIRMATORY_TEST":
        raise RuntimeError("ARC protocol is not frozen before confirmatory test")

    model = protocol.get("models", {}).get("strong_pretrained_baseline", {})
    expected = {
        "model": EXPECTED_MODEL_ID,
        "revision": EXPECTED_MODEL_REVISION,
        "license": EXPECTED_LICENSE,
        "role": "strong pretrained multiple-choice comparison",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise RuntimeError(f"frozen pretrained baseline drift: {key}={model.get(key)!r}, expected {value!r}")
    if "not parameter matched" not in str(model.get("parameter_matching", "")):
        raise RuntimeError("frozen pretrained parameter-matching boundary changed")

    budget = protocol.get("training_budget", {})
    if float(budget.get("pretrained_baseline_learning_rate", -1.0)) != 2e-5:
        raise RuntimeError("frozen pretrained learning rate changed")
    if budget.get("training_seeds") != [1, 2, 3, 4, 5]:
        raise RuntimeError("frozen confirmatory seed set changed")
    if int(budget.get("epochs", 0)) != 20 or int(budget.get("batch_size", 0)) != 32:
        raise RuntimeError("frozen confirmatory training budget changed")
    if int(budget.get("model_steps", 0)) != 1:
        raise RuntimeError("frozen LAM-JEPA model_steps changed")

    return protocol, sha256_file(path)


def verify_remote_model_revision() -> dict[str, object]:
    info = HfApi().model_info(EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_REVISION, files_metadata=True)
    resolved = str(info.sha)
    if resolved != EXPECTED_MODEL_REVISION:
        raise RuntimeError(f"model revision mismatch: expected={EXPECTED_MODEL_REVISION} actual={resolved}")

    sibling_names = sorted(str(sibling.rfilename) for sibling in (info.siblings or []))
    required_files = {"config.json", "pytorch_model.bin", "spm.model", "tokenizer_config.json"}
    missing = sorted(required_files - set(sibling_names))
    if missing:
        raise RuntimeError(f"frozen DeBERTa revision is missing required files: {missing}")
    if any(name.endswith(".py") for name in sibling_names):
        raise RuntimeError("frozen DeBERTa revision unexpectedly contains Python implementation files")

    return {
        "resolved_revision": resolved,
        "repository_files": sibling_names,
        "weight_file": "pytorch_model.bin",
        "weight_format": "pytorch_pickle_bin",
        "remote_code_allowed": False,
    }


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def encode_batch(tokenizer, examples: Sequence[ARCExample], max_length: int, device: str):
    if not examples or any(len(example.choices) != 4 for example in examples):
        raise ValueError("DeBERTa ARC batches must be non-empty with exactly four choices")
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
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def load_deberta(device: str):
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version mismatch: expected={EXPECTED_TRANSFORMERS_VERSION} actual={transformers.__version__}"
        )
    if sentencepiece.__version__ != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError(
            f"sentencepiece version mismatch: expected={EXPECTED_SENTENCEPIECE_VERSION} actual={sentencepiece.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        EXPECTED_MODEL_ID,
        revision=EXPECTED_MODEL_REVISION,
        use_fast=False,
        trust_remote_code=False,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        EXPECTED_MODEL_ID,
        revision=EXPECTED_MODEL_REVISION,
        trust_remote_code=False,
        use_safetensors=False,
    ).to(device)
    config_revision = getattr(model.config, "_commit_hash", None)
    if config_revision is not None and str(config_revision) != EXPECTED_MODEL_REVISION:
        raise RuntimeError(f"loaded model config resolved unexpected revision: {config_revision}")
    return tokenizer, model


def train_deberta(
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
    tokenizer, model = load_deberta(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = 0
    start = time.perf_counter()

    model.train()
    for epoch in range(epochs):
        for batch in iter_batches(train, batch_size, seed + epoch):
            inputs, labels = encode_batch(tokenizer, batch, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=labels)
            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("frozen DeBERTa baseline produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if max_train_steps is not None and steps >= max_train_steps:
                break
        if max_train_steps is not None and steps >= max_train_steps:
            break

    wall_seconds = float(time.perf_counter() - start)
    if steps < 1 or wall_seconds <= 0.0:
        raise RuntimeError("frozen DeBERTa baseline did not execute measurable training")
    return tokenizer, model.eval(), steps, wall_seconds


@torch.no_grad()
def predict_deberta(tokenizer, model, examples, *, batch_size: int, max_length: int, device: str):
    start = time.perf_counter()
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for offset in range(0, len(examples), batch_size):
        batch = list(examples[offset : offset + batch_size])
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
    wall_seconds = float(time.perf_counter() - start)
    return torch.cat(all_probabilities), torch.cat(all_labels), rows, wall_seconds


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen protocol-v1 DeBERTa ARC baseline smoke.")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.model_steps, args.max_train_steps or 1) < 1 or args.max_length < 8:
        parser.error("invalid non-positive execution argument")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        parser.error("development smoke requires at least two unique seeds")

    protocol, protocol_sha256 = load_and_validate_protocol(args.protocol)
    frozen_lr = float(protocol["training_budget"]["pretrained_baseline_learning_rate"])
    if args.learning_rate != frozen_lr:
        parser.error(f"learning rate must match frozen protocol: {frozen_lr}")
    if args.model_steps != int(protocol["training_budget"]["model_steps"]):
        parser.error("LAM model_steps must match frozen protocol")

    remote_model = verify_remote_model_revision()
    train_all = load_arc_split(args.train)
    validation_all = load_arc_split(args.validation)
    train = list(train_all[: args.train_limit] if args.train_limit else train_all)
    validation = list(validation_all[: args.validation_limit] if args.validation_limit else validation_all)
    if not train or not validation:
        parser.error("train and validation splits must be non-empty")
    if any(len(example.choices) != 4 for example in train + validation):
        parser.error("current ARC protocol requires exactly four choices")
    overlap = {example.item_id for example in train} & {example.item_id for example in validation}
    if overlap:
        raise SystemExit(f"ARC train/validation leakage detected: {sorted(overlap)[:5]}")

    cfg = LAMJEPAConfig()
    reversed_validation = [reverse_choices(example) for example in validation]
    records: list[dict[str, object]] = []
    lam_accuracy: list[float] = []
    deberta_accuracy: list[float] = []
    deltas: list[float] = []
    total_parameters: int | None = None
    trainable_parameters: int | None = None

    for seed in seeds:
        tokenizer, deberta, training_steps, training_seconds = train_deberta(
            train,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_train_steps=args.max_train_steps,
            device=args.device,
        )
        total, trainable_count = parameter_counts(deberta)
        if total_parameters is None:
            total_parameters = total
            trainable_parameters = trainable_count
        elif total != total_parameters or trainable_count != trainable_parameters:
            raise RuntimeError("DeBERTa parameter counts changed across seeds")

        lam_start = time.perf_counter()
        lam = _train_lam_jepa(
            train,
            cfg=cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=float(protocol["training_budget"]["lam_jepa_learning_rate"]),
            model_steps=args.model_steps,
            device=args.device,
        )
        lam_training_seconds = float(time.perf_counter() - lam_start)

        deberta_probs, deberta_labels, deberta_rows, deberta_eval_seconds = predict_deberta(
            tokenizer,
            deberta,
            validation,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        lam_eval_start = time.perf_counter()
        lam_probs, lam_labels, lam_rows = _predict_lam(
            lam,
            validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        lam_eval_seconds = float(time.perf_counter() - lam_eval_start)
        if not torch.equal(deberta_labels, lam_labels):
            raise RuntimeError("LAM-JEPA and frozen DeBERTa evaluated different validation labels")

        deberta_rev_probs, deberta_rev_labels, deberta_rev_rows, deberta_rev_seconds = predict_deberta(
            tokenizer,
            deberta,
            reversed_validation,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        lam_rev_start = time.perf_counter()
        lam_rev_probs, lam_rev_labels, lam_rev_rows = _predict_lam(
            lam,
            reversed_validation,
            cfg=cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        lam_rev_seconds = float(time.perf_counter() - lam_rev_start)
        if not torch.equal(deberta_rev_labels, lam_rev_labels):
            raise RuntimeError("LAM-JEPA and frozen DeBERTa evaluated different reversed labels")

        deberta_metrics = score_predictions(deberta_probs, deberta_labels)
        lam_metrics = score_predictions(lam_probs, lam_labels)
        deberta_rev_metrics = score_predictions(deberta_rev_probs, deberta_rev_labels)
        lam_rev_metrics = score_predictions(lam_rev_probs, lam_rev_labels)
        delta = float(lam_metrics["accuracy"] - deberta_metrics["accuracy"])
        deberta_accuracy.append(float(deberta_metrics["accuracy"]))
        lam_accuracy.append(float(lam_metrics["accuracy"]))
        deltas.append(delta)

        records.append(
            {
                "seed": seed,
                "frozen_deberta": {
                    "metrics": deberta_metrics,
                    "choice_reversal_metrics": deberta_rev_metrics,
                    "predictions": deberta_rows,
                    "choice_reversal_predictions": deberta_rev_rows,
                    "training_steps_executed": training_steps,
                    "training_wall_seconds": training_seconds,
                    "validation_wall_seconds": deberta_eval_seconds,
                    "choice_reversal_wall_seconds": deberta_rev_seconds,
                },
                "lam_jepa": {
                    "metrics": lam_metrics,
                    "choice_reversal_metrics": lam_rev_metrics,
                    "predictions": lam_rows,
                    "choice_reversal_predictions": lam_rev_rows,
                    "training_wall_seconds": lam_training_seconds,
                    "validation_wall_seconds": lam_eval_seconds,
                    "choice_reversal_wall_seconds": lam_rev_seconds,
                },
                "accuracy_delta_lam_minus_deberta": delta,
            }
        )
        del tokenizer, deberta, lam

    assert total_parameters is not None and trainable_parameters is not None
    payload = {
        "artifact_type": "lam-jepa frozen ARC protocol-v1 DeBERTa development smoke",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "protocol_status": protocol["status"],
            "development_smoke_only": True,
            "confirmatory_budget_executed": False,
            "test_split_accessed": False,
            "dataset": "AI2 ARC-Challenge",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_digest": dataset_digest(train),
            "validation_digest": dataset_digest(validation),
            "train_id_digest": id_digest(train),
            "validation_id_digest": id_digest(validation),
            "train_validation_overlap": 0,
            "seeds": seeds,
            "smoke_epochs": args.epochs,
            "smoke_batch_size": args.batch_size,
            "smoke_max_train_steps": args.max_train_steps,
            "pretrained_learning_rate": args.learning_rate,
            "lam_jepa_learning_rate": float(protocol["training_budget"]["lam_jepa_learning_rate"]),
            "model_steps": args.model_steps,
            "max_length": args.max_length,
            "device": args.device,
            "frozen_confirmatory_seeds": protocol["training_budget"]["training_seeds"],
            "frozen_confirmatory_epochs": protocol["training_budget"]["epochs"],
            "frozen_confirmatory_batch_size": protocol["training_budget"]["batch_size"],
            "primary_metric": protocol["metrics"]["primary"],
            "calibration_primary": protocol["metrics"]["calibration_primary"],
            "robustness_check": protocol["robustness"]["choice_order"],
            "pretrained_model_id": EXPECTED_MODEL_ID,
            "pretrained_model_revision": EXPECTED_MODEL_REVISION,
            "resolved_pretrained_revision": remote_model["resolved_revision"],
            "pretrained_model_license": EXPECTED_LICENSE,
            "pretrained_total_parameters": total_parameters,
            "pretrained_trainable_parameters": trainable_parameters,
            "pretrained_weight_file": remote_model["weight_file"],
            "pretrained_weight_format": remote_model["weight_format"],
            "trust_remote_code": False,
            "transformers_version": transformers.__version__,
            "sentencepiece_version": sentencepiece.__version__,
            "parameter_matching": protocol["models"]["strong_pretrained_baseline"]["parameter_matching"],
            "claim_boundary": (
                "This command validates implementation compatibility with the frozen protocol-v1 strong pretrained model. "
                "It is a bounded train/validation smoke only: not the five-seed 20-epoch confirmatory budget, not the ARC test result, "
                "not compute matched, not final independent reproduction, and not evidence of superiority or RESEARCH_COMPLETE."
            ),
        },
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_accuracy),
            "deberta_accuracy": summarize(deberta_accuracy),
            "paired_accuracy_delta_lam_minus_deberta": summarize(deltas),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["protocol"], indent=2))


if __name__ == "__main__":
    main()
