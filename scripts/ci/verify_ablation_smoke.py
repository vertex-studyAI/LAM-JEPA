from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lam_jepa.analysis.statistics import paired_summary, summarize_seed_runs
from lam_jepa.benchmarking.edtech_suite import ABLATION_VARIANTS, EDTECH_TASKS
from lam_jepa.benchmarking.evaluation_sampling import TARGET_SEMANTICS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify paired LAM-JEPA ablation evidence.")
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(args.ablation.is_file(), f"ablation evidence not found: {args.ablation}")
    payload = json.loads(args.ablation.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "ablation evidence must be a JSON object")

    protocol = payload.get("protocol")
    semantics = payload.get("target_semantics")
    digests = payload.get("sample_digests")
    variants = payload.get("variants")
    paired_effects = payload.get("paired_effects")

    require(isinstance(protocol, dict), "ablation protocol missing")
    require(semantics == TARGET_SEMANTICS, "target semantics mismatch")
    require(isinstance(digests, dict) and set(digests) == set(EDTECH_TASKS), "sample digest coverage mismatch")
    require(isinstance(variants, dict) and tuple(variants) == ABLATION_VARIANTS, "ablation variant registry mismatch")
    require(isinstance(paired_effects, dict), "paired effect summaries missing")
    require(protocol.get("variants") == list(ABLATION_VARIANTS), "protocol variant registry mismatch")
    require(protocol.get("tasks") == list(EDTECH_TASKS), "protocol task registry mismatch")
    require(
        protocol.get("pairing") == "same training seeds and identical ordered evaluation rows across variants",
        "paired ablation contract missing or weakened",
    )

    seeds = protocol.get("training_seeds")
    batch_size = protocol.get("batch_size")
    eval_batches = protocol.get("eval_batches")
    evaluation_seed = protocol.get("evaluation_seed")
    require(isinstance(seeds, list) and len(seeds) >= 2, "at least two training seeds are required")
    require(len(set(seeds)) == len(seeds), "training seeds must be unique")
    require(isinstance(batch_size, int) and batch_size >= 1, "invalid batch size")
    require(isinstance(eval_batches, int) and eval_batches >= 1, "invalid evaluation batch count")
    require(isinstance(evaluation_seed, int), "invalid evaluation seed")
    expected_n = batch_size * eval_batches

    for task in EDTECH_TASKS:
        digest = digests[task]
        require(isinstance(digest, str) and len(digest) == 64, f"{task}: invalid canonical sample digest")

    collected: dict[str, dict[str, list[float]]] = {
        variant: {task: [] for task in EDTECH_TASKS}
        for variant in ABLATION_VARIANTS
    }

    for variant in ABLATION_VARIANTS:
        variant_payload = variants[variant]
        require(isinstance(variant_payload, dict), f"{variant}: payload must be an object")
        records = variant_payload.get("records")
        aggregate = variant_payload.get("aggregate")
        require(isinstance(records, list) and len(records) == len(seeds), f"{variant}: seed record count mismatch")
        require(isinstance(aggregate, dict) and set(aggregate) == set(EDTECH_TASKS), f"{variant}: aggregate coverage mismatch")

        for expected_seed, record in zip(seeds, records, strict=True):
            require(isinstance(record, dict), f"{variant}: seed record must be an object")
            require(record.get("training_seed") == expected_seed, f"{variant}: training seed order mismatch")
            require(record.get("evaluation_seed") == evaluation_seed, f"{variant}: evaluation seed mismatch")
            scores = record.get("scores")
            require(isinstance(scores, dict) and set(scores) == set(EDTECH_TASKS), f"{variant}: task coverage mismatch")

            for task in EDTECH_TASKS:
                row = scores[task]
                require(isinstance(row, dict), f"{variant}/{task}: score row must be an object")
                require(row.get("n") == expected_n, f"{variant}/{task}: sample count mismatch")
                require(row.get("target_semantics") == TARGET_SEMANTICS[task], f"{variant}/{task}: target semantics mismatch")
                require(row.get("sample_digest") == digests[task], f"{variant}/{task}: evaluation rows are not paired")
                accuracy = float(row["accuracy"])
                confidence = float(row["confidence"])
                require(math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0, f"{variant}/{task}: invalid accuracy")
                require(math.isfinite(confidence) and 0.0 <= confidence <= 1.0, f"{variant}/{task}: invalid confidence")
                collected[variant][task].append(accuracy)

        recomputed = summarize_seed_runs(collected[variant])
        for task in EDTECH_TASKS:
            actual = aggregate[task]
            expected = recomputed[task]
            require(isinstance(actual, dict), f"{variant}/{task}: aggregate row must be an object")
            require(actual.get("n") == len(seeds), f"{variant}/{task}: aggregate seed count mismatch")
            require(close(float(actual["mean"]), float(expected["mean"])), f"{variant}/{task}: aggregate mean mismatch")
            require(close(float(actual["std"]), float(expected["std"])), f"{variant}/{task}: aggregate std mismatch")
            require(len(actual.get("ci95", [])) == 2, f"{variant}/{task}: aggregate CI missing")
            require(close(float(actual["ci95"][0]), float(expected["ci95"][0])), f"{variant}/{task}: CI lower mismatch")
            require(close(float(actual["ci95"][1]), float(expected["ci95"][1])), f"{variant}/{task}: CI upper mismatch")

    expected_effect_variants = set(ABLATION_VARIANTS) - {"full"}
    require(set(paired_effects) == expected_effect_variants, "paired effect variant coverage mismatch")
    for variant in ABLATION_VARIANTS:
        if variant == "full":
            continue
        effects = paired_effects[variant]
        require(isinstance(effects, dict) and set(effects) == set(EDTECH_TASKS), f"{variant}: paired effect task coverage mismatch")
        for task in EDTECH_TASKS:
            actual = effects[task]
            expected = paired_summary(collected["full"][task], collected[variant][task])
            expected_values = {
                "mean_full": expected.mean_a,
                "mean_variant": expected.mean_b,
                "mean_full_minus_variant": expected.mean_diff,
                "std_paired_difference": expected.std_diff,
                "cohen_d_paired": expected.cohen_d,
                "ci95_low": expected.ci_low,
                "ci95_high": expected.ci_high,
                "paired_permutation_p": expected.p_value,
            }
            require(isinstance(actual, dict), f"{variant}/{task}: paired effect row must be an object")
            for key, value in expected_values.items():
                require(key in actual, f"{variant}/{task}: missing paired effect field {key}")
                require(math.isfinite(float(actual[key])), f"{variant}/{task}: non-finite paired effect field {key}")
                require(close(float(actual[key]), float(value)), f"{variant}/{task}: paired effect mismatch for {key}")

    claim_boundary = protocol.get("claim_boundary")
    require(isinstance(claim_boundary, str), "claim boundary missing")
    for phrase in ("descriptive", "training budget", "data", "statistical power"):
        require(phrase in claim_boundary, f"claim boundary missing phrase: {phrase}")

    report = {
        "status": "passed",
        "training_seeds": seeds,
        "variants_verified": list(ABLATION_VARIANTS),
        "tasks_verified": list(EDTECH_TASKS),
        "sample_digests": digests,
        "pairing": protocol["pairing"],
        "claim_boundary": claim_boundary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
