from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split
from lam_jepa.benchmarking.arc_protocol import ARC_PROTOCOL_CHOICE_COUNT, select_protocol_eligible_examples


def split_report(path: Path) -> dict[str, object]:
    examples = load_arc_split(path)
    result = select_protocol_eligible_examples(examples)
    excluded_rows = [
        {
            "id": example.item_id,
            "choice_count": len(example.choices),
        }
        for example in result.excluded
    ]
    return {
        "source_path": str(path),
        "source_rows": result.original_count,
        "source_dataset_digest": dataset_digest(examples),
        "source_id_digest": id_digest(examples),
        "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
        "choice_count_distribution": {str(key): value for key, value in result.choice_count_distribution.items()},
        "eligible_rows": result.eligible_count,
        "excluded_rows_count": result.excluded_count,
        "eligible_dataset_digest": dataset_digest(result.eligible),
        "eligible_id_digest": result.eligible_id_digest,
        "excluded_id_digest": result.excluded_id_digest,
        "excluded_rows": excluded_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen ARC protocol-v3 exactly-four-choice eligibility rule.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.train, args.validation):
        if not path.is_file():
            parser.error(f"split not found: {path}")

    report = {
        "status": "passed",
        "protocol_id": "lam-jepa-arc-challenge-v3",
        "eligibility_rule": "retain a row if and only if len(choices) == 4",
        "decision_basis": "question structure only; labels and model outputs are not eligibility inputs",
        "test_split_accessed": False,
        "splits": {
            "train": split_report(args.train),
            "validation": split_report(args.validation),
        },
    }
    for split, block in report["splits"].items():
        if int(block["eligible_rows"]) <= 0:
            raise SystemExit(f"{split}: eligibility rule retained zero rows")
        if int(block["source_rows"]) != int(block["eligible_rows"]) + int(block["excluded_rows_count"]):
            raise SystemExit(f"{split}: eligibility partition count mismatch")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
