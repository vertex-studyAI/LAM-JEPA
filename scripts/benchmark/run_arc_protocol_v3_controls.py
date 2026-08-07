from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import (
    ARCExample,
    _train_lam_jepa,
    batchify,
    dataset_digest,
    id_digest,
    iter_minibatches,
    load_arc_split,
    reverse_choices,
    score_predictions,
)
from lam_jepa.benchmarking.arc_protocol import ARC_PROTOCOL_CHOICE_COUNT, select_protocol_eligible_examples
from lam_jepa.model import LAMJEPAConfig


PROTOCOL_ID = "lam-jepa-arc-challenge-v3"
REQUIRED_VARIANTS = ("full", "no_planner", "no_target")
NEGATIVE_CONTROL_SEED = 20260807
NEGATIVE_CONTROL_MAX_ACCURACY = 0.35


def variant_config(variant: str) -> LAMJEPAConfig:
    cfg = LAMJEPAConfig()
    if variant == "full":
        return cfg
    if variant == "no_planner":
        return replace(cfg, use_planner=False)
    if variant == "no_target":
        return replace(cfg, use_target=False)
    raise ValueError(f"unsupported protocol-v3 ARC variant: {variant}")


def label_digest(examples: Sequence[ARCExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.item_id.encode("utf-8"))
        digest.update(b":")
        digest.update(str(example.label).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def permute_training_labels(examples: Sequence[ARCExample], *, seed: int) -> list[ARCExample]:
    labels = [example.label for example in examples]
    permuted = list(labels)
    random.Random(seed).shuffle(permuted)
    if len(permuted) > 1 and permuted == labels:
        permuted = permuted[1:] + permuted[:1]
    if sorted(permuted) != sorted(labels):
        raise RuntimeError("negative-control permutation changed the training-label multiset")
    if len(permuted) > 1 and all(left == right for left, right in zip(labels, permuted, strict=True)):
        raise RuntimeError("negative-control permutation changed zero training labels")
    return [
        ARCExample(
            item_id=example.item_id,
            question=example.question,
            choices=example.choices,
            label=label,
        )
        for example, label in zip(examples, permuted, strict=True)
    ]


def eligibility_evidence(source: Sequence[ARCExample], used: Sequence[ARCExample]) -> dict[str, object]:
    partition = select_protocol_eligible_examples(source)
    used_ids = [example.item_id for example in used]
    eligible_ids = [example.item_id for example in partition.eligible]
    if used_ids != eligible_ids[: len(used_ids)]:
        raise RuntimeError("benchmark limit was not applied after the frozen eligibility rule")
    return {
        "source_rows": partition.original_count,
        "source_dataset_digest": dataset_digest(source),
        "source_id_digest": id_digest(source),
        "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
        "choice_count_distribution": {str(key): value for key, value in partition.choice_count_distribution.items()},
        "eligible_rows": partition.eligible_count,
        "eligible_dataset_digest": dataset_digest(partition.eligible),
        "eligible_id_digest": partition.eligible_id_digest,
        "excluded_rows": partition.excluded_count,
        "excluded_id_digest": partition.excluded_id_digest,
        "excluded": [
            {"id": example.item_id, "choice_count": len(example.choices)}
            for example in partition.excluded
        ],
        "used_rows": len(used),
        "used_dataset_digest": dataset_digest(used),
        "used_id_digest": id_digest(used),
    }


@torch.no_grad()
def predict_variant(
    model,
    examples: Sequence[ARCExample],
    *,
    cfg: LAMJEPAConfig,
    batch_size: int,
    model_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]], list[int]]:
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    action_steps: list[int] = []
    expected_actions = model_steps if cfg.use_planner else 0

    for start in range(0, len(examples), batch_size):
        batch = list(examples[start : start + batch_size])
        tokens, numeric_x, labels = batchify(batch, vocab_size=cfg.vocab_size, device=device)
        logits, outputs = model(tokens, numeric_x, model_steps=model_steps, deterministic=True)
        observed_actions = len(outputs["actions"])
        if observed_actions != expected_actions:
            raise RuntimeError(
                f"planner execution mismatch: use_planner={cfg.use_planner} "
                f"expected={expected_actions} observed={observed_actions}"
            )
        action_steps.append(observed_actions)
        probabilities = torch.softmax(logits, dim=-1).cpu()
        labels_cpu = labels.cpu()
        all_probs.append(probabilities)
        all_labels.append(labels_cpu)
        for example, probability, label in zip(batch, probabilities, labels_cpu, strict=True):
            rows.append(
                {
                    "id": example.item_id,
                    "label": int(label.item()),
                    "prediction": int(probability.argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                }
            )

    return torch.cat(all_probs), torch.cat(all_labels), rows, action_steps


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def paired_bootstrap_ci(deltas: Sequence[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    n = len(deltas)
    boot = [float(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))) for _ in range(samples)]
    boot.sort()
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def require_same_rows(left: list[dict[str, object]], right: list[dict[str, object]], label: str) -> None:
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise RuntimeError(f"{label}: evaluation row identity/order mismatch")
    if [row["label"] for row in left] != [row["label"] for row in right]:
        raise RuntimeError(f"{label}: evaluation labels mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eligible real-ARC protocol-v3 negative control and required ablations.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-steps", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.model_steps) < 1:
        parser.error("epochs, batch size, and model steps must all be >=1")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("development control smoke requires at least two unique seeds")

    train_source = load_arc_split(args.train)
    validation_source = load_arc_split(args.validation)
    train_partition = select_protocol_eligible_examples(train_source)
    validation_partition = select_protocol_eligible_examples(validation_source)
    train_eligible = list(train_partition.eligible)
    validation_eligible = list(validation_partition.eligible)
    train = train_eligible[: args.train_limit] if args.train_limit else train_eligible
    validation = validation_eligible[: args.validation_limit] if args.validation_limit else validation_eligible
    if not train or not validation:
        parser.error("eligible train and validation must be non-empty")
    if any(len(example.choices) != ARC_PROTOCOL_CHOICE_COUNT for example in train + validation):
        raise RuntimeError("frozen eligibility rule admitted a non-four-choice row")
    overlap = sorted({example.item_id for example in train_eligible} & {example.item_id for example in validation_eligible})
    if overlap:
        raise SystemExit(f"eligible ARC train/validation ID leakage detected: {overlap[:5]}")

    train_eligibility = eligibility_evidence(train_source, train)
    validation_eligibility = eligibility_evidence(validation_source, validation)
    expected_ids = [example.item_id for example in validation]
    reversed_validation = [reverse_choices(example) for example in validation]
    permuted_train = permute_training_labels(train, seed=NEGATIVE_CONTROL_SEED)
    original_label_digest = label_digest(train)
    permuted_label_digest = label_digest(permuted_train)
    if original_label_digest == permuted_label_digest:
        raise RuntimeError("negative-control label digest did not change")

    variant_records: dict[str, list[dict[str, object]]] = {variant: [] for variant in REQUIRED_VARIANTS}
    variant_accuracies: dict[str, list[float]] = {variant: [] for variant in REQUIRED_VARIANTS}
    negative_records: list[dict[str, object]] = []
    negative_accuracies: list[float] = []

    for seed in seeds:
        canonical_rows: list[dict[str, object]] | None = None
        for variant in REQUIRED_VARIANTS:
            cfg = variant_config(variant)
            model = _train_lam_jepa(
                train,
                cfg=cfg,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.learning_rate,
                model_steps=args.model_steps,
                device=args.device,
            )
            probs, labels, rows, action_steps = predict_variant(
                model,
                validation,
                cfg=cfg,
                batch_size=args.batch_size,
                model_steps=args.model_steps,
                device=args.device,
            )
            rev_probs, rev_labels, rev_rows, rev_action_steps = predict_variant(
                model,
                reversed_validation,
                cfg=cfg,
                batch_size=args.batch_size,
                model_steps=args.model_steps,
                device=args.device,
            )
            if [row["id"] for row in rows] != expected_ids or [row["id"] for row in rev_rows] != expected_ids:
                raise RuntimeError(f"{variant}/seed={seed}: eligible validation order changed")
            if canonical_rows is None:
                canonical_rows = rows
            else:
                require_same_rows(canonical_rows, rows, f"{variant}/seed={seed}")
            if [int(row["label"]) for row in rev_rows] != [3 - int(row["label"]) for row in rows]:
                raise RuntimeError(f"{variant}/seed={seed}: choice-reversal label remapping failed")
            metrics = score_predictions(probs, labels)
            rev_metrics = score_predictions(rev_probs, rev_labels)
            variant_accuracies[variant].append(float(metrics["accuracy"]))
            variant_records[variant].append(
                {
                    "seed": seed,
                    "use_planner": cfg.use_planner,
                    "use_target": cfg.use_target,
                    "expected_action_steps": args.model_steps if cfg.use_planner else 0,
                    "observed_action_steps": action_steps,
                    "observed_reversed_action_steps": rev_action_steps,
                    "metrics": metrics,
                    "choice_reversal_metrics": rev_metrics,
                    "predictions": rows,
                    "choice_reversal_predictions": rev_rows,
                }
            )
            del model

        negative_cfg = variant_config("full")
        negative_model = _train_lam_jepa(
            permuted_train,
            cfg=negative_cfg,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            model_steps=args.model_steps,
            device=args.device,
        )
        neg_probs, neg_labels, neg_rows, neg_action_steps = predict_variant(
            negative_model,
            validation,
            cfg=negative_cfg,
            batch_size=args.batch_size,
            model_steps=args.model_steps,
            device=args.device,
        )
        if [row["id"] for row in neg_rows] != expected_ids:
            raise RuntimeError(f"negative-control/seed={seed}: eligible validation order changed")
        require_same_rows(variant_records["full"][-1]["predictions"], neg_rows, f"negative-control/seed={seed}")
        neg_metrics = score_predictions(neg_probs, neg_labels)
        negative_accuracies.append(float(neg_metrics["accuracy"]))
        negative_records.append(
            {
                "seed": seed,
                "metrics": neg_metrics,
                "predictions": neg_rows,
                "observed_action_steps": neg_action_steps,
            }
        )
        del negative_model

    paired_effects: dict[str, dict[str, object]] = {}
    for offset, variant in enumerate(("no_planner", "no_target")):
        deltas = [
            full - ablated
            for full, ablated in zip(variant_accuracies["full"], variant_accuracies[variant], strict=True)
        ]
        ci_low, ci_high = paired_bootstrap_ci(deltas, seed=NEGATIVE_CONTROL_SEED + offset)
        mean_delta = float(statistics.fmean(deltas))
        paired_effects[variant] = {
            "seed_level_full_minus_ablation": deltas,
            "mean_full_minus_ablation": mean_delta,
            "std_paired_difference": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "observed_mechanism_numeric_criterion_met": bool(mean_delta >= 0.01 and ci_low > 0.0),
        }

    negative_summary = summarize(negative_accuracies)
    negative_pass = float(negative_summary["mean"]) <= NEGATIVE_CONTROL_MAX_ACCURACY

    payload = {
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "dataset": "AI2 ARC-Challenge",
            "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
            "eligibility_rule": "retain a row if and only if len(choices) == 4",
            "train_eligibility": train_eligibility,
            "validation_eligibility": validation_eligibility,
            "eligible_train_validation_overlap": 0,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_steps": args.model_steps,
            "required_variants": list(REQUIRED_VARIANTS),
            "negative_control_type": "deterministic training-label permutation",
            "negative_control_seed": NEGATIVE_CONTROL_SEED,
            "negative_control_max_validation_accuracy": NEGATIVE_CONTROL_MAX_ACCURACY,
            "original_training_label_digest": original_label_digest,
            "permuted_training_label_digest": permuted_label_digest,
            "test_split_policy": "not downloaded or evaluated by this development command",
            "claim_boundary": (
                "This is a bounded eligible-train/eligible-validation development smoke for frozen protocol-v3 controls only. "
                "It is not the final five-seed/20-epoch protocol, not a locked-test result, and not sufficient for a "
                "mechanism, superiority, external-validity, or RESEARCH_COMPLETE claim."
            ),
        },
        "variants": {
            variant: {
                "records": variant_records[variant],
                "accuracy": summarize(variant_accuracies[variant]),
            }
            for variant in REQUIRED_VARIANTS
        },
        "paired_effects": paired_effects,
        "negative_control": {
            "records": negative_records,
            "accuracy": negative_summary,
            "pass": negative_pass,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "variant_accuracy": {name: value["accuracy"] for name, value in payload["variants"].items()},
                "paired_effects": payload["paired_effects"],
                "negative_control": payload["negative_control"]["accuracy"] | {"pass": negative_pass},
            },
            indent=2,
        )
    )
    if not negative_pass:
        raise RuntimeError(
            "protocol-v3 negative control failed: shuffled-label mean eligible-validation accuracy "
            f"{negative_summary['mean']:.6f} > {NEGATIVE_CONTROL_MAX_ACCURACY:.2f}; investigate leakage/shortcut behavior"
        )


if __name__ == "__main__":
    main()
