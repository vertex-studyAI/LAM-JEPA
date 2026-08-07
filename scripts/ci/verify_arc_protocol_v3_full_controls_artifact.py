from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "ci"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import verify_arc_protocol_v3_controls as base


FLOAT_TOLERANCE = 1e-6
EXPECTED_SEEDS = [1, 2, 3, 4, 5]
EXPECTED_EPOCHS = 20
EXPECTED_BATCH_SIZE = 32
EXPECTED_LR = 0.0003
EXPECTED_MODEL_STEPS = 1
EXPECTED_TRAIN_ELIGIBLE = 1117
EXPECTED_VALIDATION_ELIGIBLE = 295
EXPECTED_SOURCE_RUN_ID = 31195682685
EXPECTED_SOURCE_ARTIFACT_ID = 9000793334
EXPECTED_SOURCE_HEAD_SHA = "a61f689ae1127140b960173c6f0f316862fb00f2"
EXPECTED_SOURCE_ARTIFACT_DIGEST = "sha256:55fd44d8c6633e21b3f195cc8ba27656c0ddc8a2fe9e96ec8756a12c035bb295"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(left: float, right: float, *, label: str, tolerance: float = FLOAT_TOLERANCE) -> None:
    require(
        math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance),
        f"{label}: mismatch: declared={left!r} recomputed={right!r}",
    )


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def verify_summary(declared: object, expected: dict[str, float | int], label: str) -> None:
    require(isinstance(declared, dict), f"{label}: summary missing")
    require(int(declared.get("n", -1)) == int(expected["n"]), f"{label}: n mismatch")
    close(float(declared.get("mean", float("nan"))), float(expected["mean"]), label=f"{label}: mean")
    close(float(declared.get("std", float("nan"))), float(expected["std"]), label=f"{label}: std")


def paired_bootstrap_ci(deltas: list[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    draws = [float(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))) for _ in range(samples)]
    draws.sort()
    return draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]


def verify_numeric_list(declared: object, expected: list[float], label: str) -> None:
    require(isinstance(declared, list) and len(declared) == len(expected), f"{label}: length mismatch")
    for index, (left, right) in enumerate(zip(declared, expected, strict=True)):
        close(float(left), float(right), label=f"{label}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the retained frozen-budget ARC v3 controls artifact without retraining.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--artifact-metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    metadata = json.loads(args.artifact_metadata.read_text(encoding="utf-8"))

    require(metadata.get("id") == EXPECTED_SOURCE_ARTIFACT_ID, "source artifact id mismatch")
    require(metadata.get("name") == "arc-protocol-v3-full-controls-validation", "source artifact name mismatch")
    require(metadata.get("digest") == EXPECTED_SOURCE_ARTIFACT_DIGEST, "source artifact digest mismatch")
    workflow_run = metadata.get("workflow_run") or {}
    require(workflow_run.get("id") == EXPECTED_SOURCE_RUN_ID, "source workflow run id mismatch")
    require(workflow_run.get("head_sha") == EXPECTED_SOURCE_HEAD_SHA, "source workflow head SHA mismatch")
    require(metadata.get("expired") is False, "source artifact expired")

    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen protocol")
    require(frozen.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol v3 is not frozen")
    eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 choice-count drift")

    protocol = payload.get("protocol") or {}
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol mismatch")
    require(protocol.get("dataset") == "AI2 ARC-Challenge", "result dataset mismatch")
    require(protocol.get("required_choice_count") == required_choice_count, "result required choice count mismatch")
    require(protocol.get("eligibility_rule") == eligibility.get("rule"), "result eligibility rule mismatch")
    require(protocol.get("seeds") == EXPECTED_SEEDS, "frozen seed budget mismatch")
    require(int(protocol.get("epochs", 0)) == EXPECTED_EPOCHS, "frozen epoch budget mismatch")
    require(int(protocol.get("batch_size", 0)) == EXPECTED_BATCH_SIZE, "frozen batch-size mismatch")
    close(float(protocol.get("learning_rate", 0.0)), EXPECTED_LR, label="frozen learning rate", tolerance=1e-12)
    require(int(protocol.get("model_steps", 0)) == EXPECTED_MODEL_STEPS, "frozen planner-step mismatch")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "locked-test boundary weakened")
    require(protocol.get("eligible_train_validation_overlap") == 0, "declared eligible train/validation leakage")

    train_eligible, train_used = base.independently_verify_eligibility(
        args.train,
        protocol.get("train_eligibility"),
        required_choice_count=required_choice_count,
    )
    validation_eligible, validation_used = base.independently_verify_eligibility(
        args.validation,
        protocol.get("validation_eligibility"),
        required_choice_count=required_choice_count,
    )
    require(len(train_eligible) == len(train_used) == EXPECTED_TRAIN_ELIGIBLE, "full run did not use all 1117 eligible train rows")
    require(len(validation_eligible) == len(validation_used) == EXPECTED_VALIDATION_ELIGIBLE, "full run did not use all 295 eligible validation rows")
    require(not ({row.item_id for row in train_eligible} & {row.item_id for row in validation_eligible}), "independent eligible train/validation leakage")

    canonical_ids = [row.item_id for row in validation_used]
    canonical_labels = [row.label for row in validation_used]
    model_steps = int(protocol["model_steps"])
    variants = payload.get("variants")
    require(isinstance(variants, dict) and set(variants) == {"full", "no_planner", "no_target"}, "variant payload mismatch")

    accuracies: dict[str, list[float]] = {}
    for variant in ("full", "no_planner", "no_target"):
        block = variants[variant]
        records = block.get("records")
        require(isinstance(records, list) and len(records) == len(EXPECTED_SEEDS), f"{variant}: five seed records required")
        values: list[float] = []
        for expected_seed, record in zip(EXPECTED_SEEDS, records, strict=True):
            require(record.get("seed") == expected_seed, f"{variant}: seed mismatch")
            expected_use_planner = variant != "no_planner"
            expected_use_target = variant != "no_target"
            require(record.get("use_planner") is expected_use_planner, f"{variant}: planner flag mismatch")
            require(record.get("use_target") is expected_use_target, f"{variant}: target flag mismatch")
            expected_actions = model_steps if expected_use_planner else 0
            require(record.get("expected_action_steps") == expected_actions, f"{variant}: expected action count mismatch")
            for key in ("observed_action_steps", "observed_reversed_action_steps"):
                observed = record.get(key)
                require(isinstance(observed, list) and observed and all(step == expected_actions for step in observed), f"{variant}: {key} mismatch")

            recomputed, ids, labels = base.metrics_from_rows(record.get("predictions"))
            reversed_metrics, reversed_ids, reversed_labels = base.metrics_from_rows(record.get("choice_reversal_predictions"))
            require(ids == canonical_ids and labels == canonical_labels, f"{variant}: eligible validation pairing mismatch")
            require(reversed_ids == canonical_ids, f"{variant}: choice reversal changed item identity")
            require(reversed_labels == [3 - label for label in canonical_labels], f"{variant}: choice reversal label remapping failed")
            base.verify_metrics(record.get("metrics"), recomputed, f"{variant}/seed={expected_seed}")
            base.verify_metrics(record.get("choice_reversal_metrics"), reversed_metrics, f"{variant}/seed={expected_seed}/reversed")
            values.append(recomputed["accuracy"])
        accuracies[variant] = values
        verify_summary(block.get("accuracy"), summary(values), f"{variant}/accuracy")

    paired = payload.get("paired_effects")
    require(isinstance(paired, dict) and set(paired) == {"no_planner", "no_target"}, "paired effects missing")
    paired_verified: dict[str, dict[str, object]] = {}
    for offset, variant in enumerate(("no_planner", "no_target")):
        deltas = [full - ablated for full, ablated in zip(accuracies["full"], accuracies[variant], strict=True)]
        block = paired[variant]
        verify_numeric_list(block.get("seed_level_full_minus_ablation"), deltas, f"{variant}: paired deltas")
        mean_delta = float(statistics.fmean(deltas))
        std_delta = float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0
        ci_low, ci_high = paired_bootstrap_ci(deltas, seed=20260807 + offset)
        close(float(block.get("mean_full_minus_ablation")), mean_delta, label=f"{variant}: mean delta")
        close(float(block.get("std_paired_difference")), std_delta, label=f"{variant}: paired std")
        close(float(block.get("paired_bootstrap_ci95_low")), ci_low, label=f"{variant}: CI low")
        close(float(block.get("paired_bootstrap_ci95_high")), ci_high, label=f"{variant}: CI high")
        expected_numeric = mean_delta >= 0.01 and ci_low > 0.0
        require(block.get("observed_mechanism_numeric_criterion_met") is expected_numeric, f"{variant}: numeric mechanism flag mismatch")
        paired_verified[variant] = {
            "seed_level_full_minus_ablation": deltas,
            "mean_full_minus_ablation": mean_delta,
            "std_paired_difference": std_delta,
            "paired_bootstrap_ci95_low": ci_low,
            "paired_bootstrap_ci95_high": ci_high,
            "observed_mechanism_numeric_criterion_met": expected_numeric,
        }

    negative = payload.get("negative_control")
    require(isinstance(negative, dict), "negative-control payload missing")
    negative_records = negative.get("records")
    require(isinstance(negative_records, list) and len(negative_records) == len(EXPECTED_SEEDS), "negative-control five seed records required")
    negative_values: list[float] = []
    for expected_seed, record in zip(EXPECTED_SEEDS, negative_records, strict=True):
        require(record.get("seed") == expected_seed, "negative-control seed mismatch")
        recomputed, ids, labels = base.metrics_from_rows(record.get("predictions"))
        require(ids == canonical_ids and labels == canonical_labels, "negative control changed eligible validation rows or labels")
        observed = record.get("observed_action_steps")
        require(isinstance(observed, list) and observed and all(step == model_steps for step in observed), "negative control did not exercise planner")
        base.verify_metrics(record.get("metrics"), recomputed, f"negative/seed={expected_seed}")
        negative_values.append(recomputed["accuracy"])
    negative_summary = summary(negative_values)
    verify_summary(negative.get("accuracy"), negative_summary, "negative-control/accuracy")
    negative_pass = float(negative_summary["mean"]) <= 0.35
    require(negative.get("pass") is negative_pass, "negative-control pass flag mismatch")
    require(negative_pass, f"negative-control failure: mean accuracy {negative_summary['mean']:.9f} > 0.35")

    report = {
        "verdict": "PROTOCOL_V3_FULL_CONTROLS_ARTIFACT_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "source_evidence": {
            "workflow_run_id": EXPECTED_SOURCE_RUN_ID,
            "artifact_id": EXPECTED_SOURCE_ARTIFACT_ID,
            "head_sha": EXPECTED_SOURCE_HEAD_SHA,
            "artifact_digest": EXPECTED_SOURCE_ARTIFACT_DIGEST,
        },
        "budget": {
            "seeds": EXPECTED_SEEDS,
            "epochs": EXPECTED_EPOCHS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "learning_rate": EXPECTED_LR,
            "model_steps": EXPECTED_MODEL_STEPS,
            "train_eligible_rows": len(train_used),
            "validation_eligible_rows": len(validation_used),
        },
        "variant_accuracy": {variant: summary(values) for variant, values in accuracies.items()},
        "paired_effects": paired_verified,
        "negative_control_accuracy": negative_summary,
        "negative_control_pass": True,
        "locked_test_evaluated": False,
        "mechanism_claim_authorized": False,
        "research_complete": False,
        "floating_point_note": "Runner aggregates Torch float32 accuracies; verifier recomputes exact row fractions. Numerical comparisons use 1e-6 tolerance while row identities, labels, seeds, budgets, action counts, and digests remain exact.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
