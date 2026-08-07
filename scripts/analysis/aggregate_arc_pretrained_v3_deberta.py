from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SEEDS = [1, 2, 3, 4, 5]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def stable_protocol_view(protocol: dict[str, object]) -> dict[str, object]:
    keys = (
        "protocol_id",
        "implementation_config_id",
        "implementation_config_sha256",
        "run_stage",
        "frozen_config_seeds",
        "dataset",
        "eligibility_rule",
        "required_choice_count",
        "train_source_eligibility",
        "validation_source_eligibility",
        "train_examples",
        "validation_examples",
        "train_digest",
        "validation_digest",
        "train_id_digest",
        "validation_id_digest",
        "train_validation_overlap",
        "epochs",
        "batch_size",
        "lam_jepa_learning_rate",
        "pretrained_learning_rate",
        "max_length",
        "max_train_steps",
        "model_steps",
        "optimizer",
        "device",
        "primary_metric",
        "calibration_metrics",
        "robustness_check",
        "pretrained_model_id",
        "pretrained_model_revision",
        "resolved_pretrained_revision",
        "pretrained_model_license",
        "pretrained_model_trainable_parameters",
        "transformers_version",
        "transformers_version_pin",
        "expected_training_steps_per_seed",
        "comparison_type",
        "test_split_policy",
        "claim_boundary",
    )
    return {key: protocol.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five independently verified protocol-v3 DeBERTa validation seed shards.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--verification-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    require(len(args.inputs) == 5, f"expected five result shards, got {len(args.inputs)}")
    require(len(args.verification_inputs) == 5, f"expected five verification shards, got {len(args.verification_inputs)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    verification = [json.loads(path.read_text(encoding="utf-8")) for path in args.verification_inputs]

    by_seed: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    for payload in payloads:
        protocol = payload.get("protocol") or {}
        records = payload.get("records") or []
        require(protocol.get("run_stage") == "validation_stage", "non-validation shard supplied")
        seed = protocol.get("validation_shard_seed")
        require(isinstance(seed, int), "shard seed marker missing")
        require(protocol.get("seeds") == [seed], f"shard {seed}: executed seed list mismatch")
        require(len(records) == 1 and records[0].get("seed") == seed, f"shard {seed}: record mismatch")
        require(seed not in by_seed, f"duplicate result shard for seed {seed}")
        by_seed[seed] = (payload, {})

    verify_by_seed: dict[int, dict[str, object]] = {}
    for report in verification:
        require(report.get("verdict") == "PROTOCOL_V3_DEBERTA_VALIDATION_SHARD_VERIFIED_ONLY", "shard verification did not pass")
        require(report.get("locked_test_evaluated") is False, "shard verification reports locked-test access")
        require(report.get("research_complete") is False, "shard verification incorrectly reports research complete")
        seed = report.get("seed")
        require(isinstance(seed, int), "verification seed missing")
        require(seed not in verify_by_seed, f"duplicate verification for seed {seed}")
        verify_by_seed[seed] = report

    require(sorted(by_seed) == EXPECTED_SEEDS, f"result shards do not exactly cover frozen seeds: {sorted(by_seed)}")
    require(sorted(verify_by_seed) == EXPECTED_SEEDS, f"verification shards do not exactly cover frozen seeds: {sorted(verify_by_seed)}")

    first = by_seed[1][0]
    first_protocol = first["protocol"]
    stable = stable_protocol_view(first_protocol)
    records: list[dict[str, object]] = []
    lam_values: list[float] = []
    deberta_values: list[float] = []
    deltas: list[float] = []
    total_wall_clock = 0.0
    shard_sources: list[dict[str, object]] = []

    for seed in EXPECTED_SEEDS:
        payload = by_seed[seed][0]
        protocol = payload["protocol"]
        require(stable_protocol_view(protocol) == stable, f"seed {seed}: scientific/execution protocol differs from seed 1")
        require(protocol.get("validation_shard_seed") == seed, f"seed {seed}: shard marker mismatch")
        report = verify_by_seed[seed]
        require(report.get("seed") == seed and report.get("training_steps") == 700, f"seed {seed}: verification evidence mismatch")
        record = payload["records"][0]
        records.append(record)
        lam_accuracy = float(record["lam_jepa"]["metrics"]["accuracy"])
        deberta_accuracy = float(record["pretrained_baseline"]["metrics"]["accuracy"])
        delta = float(record["accuracy_delta_lam_minus_pretrained"])
        lam_values.append(lam_accuracy)
        deberta_values.append(deberta_accuracy)
        deltas.append(delta)
        total_wall_clock += float(protocol.get("total_wall_clock_seconds", 0.0))
        shard_sources.append(
            {
                "seed": seed,
                "result_file": str(args.inputs[EXPECTED_SEEDS.index(seed)]),
                "verification_verdict": report["verdict"],
                "training_steps": report["training_steps"],
            }
        )

    aggregate_protocol = dict(first_protocol)
    aggregate_protocol["seeds"] = list(EXPECTED_SEEDS)
    aggregate_protocol["validation_shard_seed"] = None
    aggregate_protocol["aggregation_mode"] = "five independently executed and independently verified frozen validation seed shards"
    aggregate_protocol["total_wall_clock_seconds"] = total_wall_clock
    aggregate_protocol["shard_sources"] = shard_sources

    output = {
        "protocol": aggregate_protocol,
        "records": records,
        "summary": {
            "lam_accuracy": summarize(lam_values),
            "pretrained_accuracy": summarize(deberta_values),
            "paired_accuracy_delta_lam_minus_pretrained": summarize(deltas),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": aggregate_protocol["protocol_id"], "seeds": EXPECTED_SEEDS, "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
