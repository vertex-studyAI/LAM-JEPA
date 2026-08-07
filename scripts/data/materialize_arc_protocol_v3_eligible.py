from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples


def materialize(input_path: Path, output_path: Path, split: str) -> dict[str, object]:
    source_examples = load_arc_split(input_path)
    eligibility = select_protocol_eligible_examples(source_examples)
    retained_ids = [example.item_id for example in eligibility.eligible]
    retained_id_set = set(retained_ids)
    if len(retained_ids) != len(retained_id_set):
        raise SystemExit(f"{split}: duplicate eligible IDs")

    frame = pd.read_parquet(input_path)
    source_ids = frame["id"].astype(str).tolist()
    if source_ids != [example.item_id for example in source_examples]:
        raise SystemExit(f"{split}: parquet order disagrees with validated ARC loader")
    filtered = frame[frame["id"].astype(str).isin(retained_id_set)].copy()
    if filtered["id"].astype(str).tolist() != retained_ids:
        raise SystemExit(f"{split}: eligibility materialization changed source order")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    written_examples = load_arc_split(output_path)
    if [example.item_id for example in written_examples] != retained_ids:
        raise SystemExit(f"{split}: written eligible parquet changed identity/order")
    if any(len(example.choices) != 4 for example in written_examples):
        raise SystemExit(f"{split}: written eligible parquet contains non-four-choice row")

    return {
        "protocol_id": "lam-jepa-arc-challenge-v3",
        "split": split,
        "source_rows": len(source_examples),
        "eligible_rows": len(written_examples),
        "excluded_rows": len(eligibility.excluded),
        "source_dataset_digest": dataset_digest(source_examples),
        "source_id_digest": id_digest(source_examples),
        "eligible_dataset_digest": dataset_digest(written_examples),
        "eligible_id_digest": id_digest(written_examples),
        "excluded_id_digest": id_digest(eligibility.excluded),
        "excluded": [
            {"id": example.item_id, "choice_count": len(example.choices)}
            for example in eligibility.excluded
        ],
        "preserved_source_order": True,
        "test_split_accessed": split == "test",
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ARC rows eligible under frozen protocol v3.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input split not found: {args.input}")
    report = materialize(args.input, args.output, args.split)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
