from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lam_jepa.benchmarking.arc_challenge import load_arc_split


def exclusion_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the predeclared ARC v3 four-choice structural filter.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    examples = load_arc_split(args.input)
    retained_ids = [example.item_id for example in examples if len(example.choices) == 4]
    excluded = [
        {"id": example.item_id, "choice_count": len(example.choices)}
        for example in examples
        if len(example.choices) != 4
    ]
    retained_set = set(retained_ids)
    if len(retained_set) != len(retained_ids):
        raise SystemExit("duplicate retained ARC ids")

    frame = pd.read_parquet(args.input)
    source_ids = frame["id"].astype(str).tolist()
    if source_ids != [example.item_id for example in examples]:
        raise SystemExit("source row order disagrees with validated ARC loader")
    filtered = frame[frame["id"].astype(str).isin(retained_set)].copy()
    filtered_ids = filtered["id"].astype(str).tolist()
    if filtered_ids != retained_ids:
        raise SystemExit("structural filter changed retained-row order")
    if len(filtered) + len(excluded) != len(frame):
        raise SystemExit("structural filter lost or duplicated rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    verified = load_arc_split(args.output)
    if [example.item_id for example in verified] != retained_ids:
        raise SystemExit("written filtered parquet changed row identity/order")
    if any(len(example.choices) != 4 for example in verified):
        raise SystemExit("filtered parquet still contains non-four-choice rows")

    report = {
        "protocol_id": "lam-jepa-arc-challenge-v3",
        "dataset": "AI2 ARC-Challenge",
        "split": args.split,
        "selection_rule": "retain rows with exactly four answer choices",
        "decision_basis": "choice cardinality only",
        "test_split_accessed": args.split == "test",
        "input_rows": len(frame),
        "retained_rows": len(filtered),
        "excluded_rows": excluded,
        "excluded_identity_digest": exclusion_digest(excluded),
        "preserved_source_order": True,
        "output_sha256": file_sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
