from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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

from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS, seed_sweep


CLAIM_BOUNDARY = (
    "Multi-seed descriptive results for the declared synthetic evaluation protocol only. "
    "Confidence intervals quantify seed-level uncertainty under this protocol; they do not establish benchmark validity, "
    "natural-language answer correctness for concept-proxy tasks, educational effectiveness, held-out generalization, "
    "novelty, or superiority over external systems."
)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 2:
        parser.error("--seeds requires at least two values for a multi-seed paper package")
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds values must be unique")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be at least 1")
    return seeds


def render_markdown(summary: dict, semantics: dict) -> str:
    lines = [
        "# LAM-JEPA Seed Summary",
        "",
        "| Task | Target semantics | Mean accuracy | Std. dev. | 95% bootstrap CI | Seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in EDTECH_TASKS:
        row = summary[task]
        ci_low, ci_high = row["ci95"]
        lines.append(
            f"| {task} | {semantics[task]} | {row['mean']:.6f} | {row['std']:.6f} | "
            f"[{ci_low:.6f}, {ci_high:.6f}] | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, summary: dict, semantics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "target_semantics",
                "mean_accuracy",
                "std_accuracy",
                "ci95_low",
                "ci95_high",
                "training_seeds",
            ),
        )
        writer.writeheader()
        for task in EDTECH_TASKS:
            row = summary[task]
            writer.writerow(
                {
                    "task": task,
                    "target_semantics": semantics[task],
                    "mean_accuracy": row["mean"],
                    "std_accuracy": row["std"],
                    "ci95_low": row["ci95"][0],
                    "ci95_high": row["ci95"][1],
                    "training_seeds": row["n"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible paper-results artifact package.")
    parser.add_argument("--out-dir", type=Path, default=Path("papers"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--evaluation-seed", type=int, default=1007)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--training-task", type=str, default="mixed")
    args = parser.parse_args()
    seeds = validate_args(args, parser)

    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    supplementary_dir = out_dir / "supplementary"
    tables_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    payload = seed_sweep(
        seeds=seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        task=args.training_task,
        eval_batches=args.eval_batches,
        evaluation_seed=args.evaluation_seed,
    )

    summary_json = tables_dir / "seed_summary.json"
    summary_csv = tables_dir / "seed_summary.csv"
    summary_markdown = tables_dir / "seed_summary.md"
    records_json = supplementary_dir / "seed_records.json"
    digests_json = supplementary_dir / "evaluation_sample_digests.json"
    manifest_json = out_dir / "manifest.json"

    summary_json.write_text(json.dumps(payload["aggregate"], indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, payload["aggregate"], payload["target_semantics"])
    summary_markdown.write_text(
        render_markdown(payload["aggregate"], payload["target_semantics"]),
        encoding="utf-8",
    )
    records_json.write_text(json.dumps(payload["records"], indent=2) + "\n", encoding="utf-8")
    digests_json.write_text(json.dumps(payload["sample_digests"], indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "artifact_type": "lam-jepa multi-seed paper-results package",
        "protocol": payload["protocol"],
        "target_semantics": payload["target_semantics"],
        "files": {
            "summary_json": str(summary_json.relative_to(out_dir)),
            "summary_csv": str(summary_csv.relative_to(out_dir)),
            "summary_markdown": str(summary_markdown.relative_to(out_dir)),
            "seed_records": str(records_json.relative_to(out_dir)),
            "evaluation_sample_digests": str(digests_json.relative_to(out_dir)),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
