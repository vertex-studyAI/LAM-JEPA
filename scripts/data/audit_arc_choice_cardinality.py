from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from lam_jepa.benchmarking.arc_challenge import load_arc_split


def summarize(path: Path) -> dict:
    examples = load_arc_split(path)
    counts = Counter(len(example.choices) for example in examples)
    non_four = [
        {"id": example.item_id, "choice_count": len(example.choices)}
        for example in examples
        if len(example.choices) != 4
    ]
    digest = hashlib.sha256()
    for row in non_four:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "rows": len(examples),
        "choice_count_distribution": {str(key): value for key, value in sorted(counts.items())},
        "four_choice_rows": counts.get(4, 0),
        "non_four_choice_rows": len(non_four),
        "non_four_choice_identity_digest": digest.hexdigest(),
        "non_four_choice_rows": non_four,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ARC train/validation answer-choice cardinality without accessing test.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "dataset": "AI2 ARC-Challenge",
        "test_split_accessed": False,
        "train": summarize(args.train),
        "validation": summarize(args.validation),
        "interpretation": (
            "If any non-four-choice rows exist, the current fixed four-class LAMARCClassifier cannot satisfy "
            "the frozen full-split ARC protocol without either a pre-test protocol revision or a variable-choice implementation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
