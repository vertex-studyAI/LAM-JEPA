from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from lam_jepa.analysis.statistics import summarize_seed_runs
from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS
from lam_jepa.benchmarking.evaluation_sampling import TARGET_SEMANTICS


REQUIRED_CLAIM_PHRASES = (
    "seed-level uncertainty",
    "do not establish benchmark validity",
    "natural-language answer correctness for concept-proxy tasks",
    "educational effectiveness",
    "held-out generalization",
    "novelty",
    "superiority over external systems",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path, label: str) -> dict | list:
    require(path.is_file(), f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, (dict, list)), f"{label} must be JSON object or array")
    return payload


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a LAM-JEPA paper-results artifact package.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.out_dir / "manifest.json", "manifest")
    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("schema_version") == 1, "unsupported paper artifact schema")
    require(manifest.get("artifact_type") == "lam-jepa multi-seed paper-results package", "artifact type mismatch")

    protocol = manifest.get("protocol")
    files = manifest.get("files")
    semantics = manifest.get("target_semantics")
    require(isinstance(protocol, dict), "protocol missing")
    require(isinstance(files, dict), "file manifest missing")
    require(semantics == TARGET_SEMANTICS, "target semantics mismatch")
    require(protocol.get("tasks") == list(EDTECH_TASKS), "task registry mismatch")
    require(
        protocol.get("evaluation_pairing") == "identical ordered evaluation rows across training seeds",
        "evaluation pairing contract missing or weakened",
    )

    seeds = protocol.get("training_seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2, "paper package requires at least two training seeds")
    require(len(set(seeds)) == len(seeds), "training seeds must be unique")
    batch_size = protocol.get("batch_size")
    eval_batches = protocol.get("eval_batches")
    evaluation_seed = protocol.get("evaluation_seed")
    require(isinstance(batch_size, int) and batch_size >= 1, "invalid batch size")
    require(isinstance(eval_batches, int) and eval_batches >= 1, "invalid eval batch count")
    require(isinstance(evaluation_seed, int), "invalid evaluation seed")
    expected_n = batch_size * eval_batches

    required_file_keys = {
        "summary_json",
        "summary_csv",
        "summary_markdown",
        "seed_records",
        "evaluation_sample_digests",
    }
    require(set(files) == required_file_keys, "paper artifact file manifest mismatch")
    resolved = {name: args.out_dir / relative for name, relative in files.items()}
    for name, path in resolved.items():
        require(path.is_file(), f"declared paper artifact missing: {name} -> {path}")

    summary = read_json(resolved["summary_json"], "seed summary")
    records = read_json(resolved["seed_records"], "seed records")
    digests = read_json(resolved["evaluation_sample_digests"], "evaluation sample digests")
    require(isinstance(summary, dict), "seed summary must be an object")
    require(isinstance(records, list), "seed records must be an array")
    require(isinstance(digests, dict), "sample digests must be an object")
    require(set(summary) == set(EDTECH_TASKS), "seed summary task coverage mismatch")
    require(set(digests) == set(EDTECH_TASKS), "sample digest task coverage mismatch")
    require(len(records) == len(seeds), "seed record count mismatch")

    per_task: dict[str, list[float]] = {task: [] for task in EDTECH_TASKS}
    for expected_seed, record in zip(seeds, records, strict=True):
        require(isinstance(record, dict), "seed record must be an object")
        require(record.get("training_seed") == expected_seed, "training seed order mismatch")
        require(record.get("evaluation_seed") == evaluation_seed, "evaluation seed mismatch")
        scores = record.get("scores")
        require(isinstance(scores, dict) and set(scores) == set(EDTECH_TASKS), "seed score coverage mismatch")

        for task in EDTECH_TASKS:
            row = scores[task]
            require(isinstance(row, dict), f"{task}: score row must be an object")
            require(row.get("n") == expected_n, f"{task}: sample count mismatch")
            require(row.get("target_semantics") == TARGET_SEMANTICS[task], f"{task}: target semantics mismatch")
            digest = row.get("sample_digest")
            require(isinstance(digest, str) and len(digest) == 64, f"{task}: invalid sample digest")
            require(digest == digests[task], f"{task}: evaluation rows changed across training seeds")
            accuracy = float(row["accuracy"])
            require(math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0, f"{task}: invalid accuracy")
            per_task[task].append(accuracy)

    recomputed = summarize_seed_runs(per_task)
    for task in EDTECH_TASKS:
        actual = summary[task]
        expected = recomputed[task]
        require(isinstance(actual, dict), f"{task}: summary row must be an object")
        require(actual.get("n") == len(seeds), f"{task}: seed count mismatch")
        require(close(float(actual["mean"]), float(expected["mean"])), f"{task}: mean mismatch")
        require(close(float(actual["std"]), float(expected["std"])), f"{task}: std mismatch")
        require(len(actual.get("ci95", [])) == 2, f"{task}: invalid confidence interval")
        require(close(float(actual["ci95"][0]), float(expected["ci95"][0])), f"{task}: CI lower mismatch")
        require(close(float(actual["ci95"][1]), float(expected["ci95"][1])), f"{task}: CI upper mismatch")

    with resolved["summary_csv"].open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require([row.get("task") for row in csv_rows] == list(EDTECH_TASKS), "CSV task order mismatch")
    for row, task in zip(csv_rows, EDTECH_TASKS, strict=True):
        require(row.get("target_semantics") == TARGET_SEMANTICS[task], f"{task}: CSV semantics mismatch")
        require(close(float(row["mean_accuracy"]), float(summary[task]["mean"])), f"{task}: CSV mean mismatch")
        require(close(float(row["std_accuracy"]), float(summary[task]["std"])), f"{task}: CSV std mismatch")
        require(int(row["training_seeds"]) == len(seeds), f"{task}: CSV seed count mismatch")

    markdown = resolved["summary_markdown"].read_text(encoding="utf-8")
    for task in EDTECH_TASKS:
        require(f"| {task} | {TARGET_SEMANTICS[task]} |" in markdown, f"{task}: missing Markdown row")

    claim_boundary = manifest.get("claim_boundary")
    require(isinstance(claim_boundary, str), "claim boundary missing")
    for phrase in REQUIRED_CLAIM_PHRASES:
        require(phrase in claim_boundary, f"claim boundary missing: {phrase}")
    require(claim_boundary in markdown, "Markdown table omits the manifest claim boundary")

    report = {
        "status": "passed",
        "schema_version": manifest["schema_version"],
        "protocol": protocol,
        "tasks_verified": len(EDTECH_TASKS),
        "seed_records_verified": len(records),
        "sample_digests": digests,
        "claim_boundary": claim_boundary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
