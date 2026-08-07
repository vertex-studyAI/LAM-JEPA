from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize rows eligible under frozen ARC protocol v3.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    examples = load_arc_split(args.input)
    selection = select_protocol_eligible_examples(examples)
    eligible_ids = [example.item_id for example in selection.eligible]
    if len(eligible_ids) != len(set(eligible_ids)):
        raise SystemExit("duplicate eligible ARC IDs")

    frame = pd.read_parquet(args.input)
    source_ids = frame["id"].astype(str).tolist()
    if source_ids != [example.item_id for example in examples]:
        raise SystemExit("validated loader order disagrees with parquet order")
    eligible_set = set(eligible_ids)
    filtered = frame[frame["id"].astype(str).isin(eligible_set)].copy()
    if filtered["id"].astype(str).tolist() != eligible_ids:
        raise SystemExit("eligibility materialization changed source order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.output, index=False)
    written = load_arc_split(args.output)
    if [example.item_id for example in written] != eligible_ids:
        raise SystemExit("written eligible split changed row identity/order")
    if any(len(example.choices) != 4 for example in written):
        raise SystemExit("written eligible split contains non-four-choice row")

    report = {
        "protocol_id": "lam-jepa-arc-challenge-v3",
        "split": args.split,
        "source_rows": len(examples),
        "eligible_rows": len(written),
        "excluded_rows": len(selection.excluded),
        "source_dataset_digest": dataset_digest(examples),
        "source_id_digest": id_digest(examples),
        "eligible_dataset_digest": dataset_digest(written),
        "eligible_id_digest": id_digest(written),
        "excluded_id_digest": id_digest(selection.excluded),
        "excluded": [{"id": e.item_id, "choice_count": len(e.choices)} for e in selection.excluded],
        "preserved_source_order": True,
        "test_split_accessed": args.split == "test",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
