from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def verify_split(block: dict, *, split: str, source_rows: int, eligible_rows: int) -> dict[str, object]:
    require(block.get("split") == split, f"{split}: wrong split")
    require(block.get("source_rows") == source_rows, f"{split}: source count drift")
    require(block.get("eligible_rows") == eligible_rows, f"{split}: eligible count drift")
    require(block.get("excluded_rows") == source_rows - eligible_rows, f"{split}: excluded count drift")
    require(block.get("max_len_whitespace_tokens") == 96, f"{split}: cutoff drift")
    rows = block.get("row_records")
    require(isinstance(rows, list) and len(rows) == eligible_rows, f"{split}: incomplete rows")
    ids = [str(row.get("id", "")) for row in rows]
    require(all(ids) and len(ids) == len(set(ids)), f"{split}: duplicate/missing ids")
    digests = [str(row.get("token_digest", "")) for row in rows]
    require(all(len(value) == 64 for value in digests), f"{split}: bad token digest")
    require(block.get("unique_token_sequences") == len(set(digests)), f"{split}: unique digest mismatch")
    expected_duplicates = [
        {"digest": digest, "count": count}
        for digest, count in Counter(digests).items()
        if count > 1
    ]
    require(block.get("duplicate_token_sequence_groups") == expected_duplicates, f"{split}: duplicate digest groups drift")

    bool_keys = (
        "choices_marker_visible",
        "any_choice_text_starts_visible",
        "all_choice_texts_start_visible",
        "all_choice_texts_fully_visible",
        "correct_choice_text_starts_visible",
        "correct_choice_text_fully_visible",
    )
    fractions = block.get("fractions")
    require(isinstance(fractions, dict), f"{split}: fractions missing")
    for key in bool_keys:
        expected = sum(bool(row[key]) for row in rows) / len(rows)
        require(math.isclose(float(fractions.get(key, -1)), expected, rel_tol=1e-12, abs_tol=1e-12), f"{split}: fraction drift for {key}")

    no_choice = [row["id"] for row in rows if not row["any_choice_text_starts_visible"]]
    correct_missing = [row["id"] for row in rows if not row["correct_choice_text_starts_visible"]]
    all_full = [row["id"] for row in rows if row["all_choice_texts_fully_visible"]]
    require(block.get("rows_with_no_choice_text_visible") == no_choice, f"{split}: no-choice list drift")
    require(block.get("rows_where_correct_choice_never_starts") == correct_missing, f"{split}: correct-choice list drift")
    require(block.get("rows_where_all_choices_fully_visible") == all_full, f"{split}: fully-visible list drift")

    labels = Counter(int(row["label"]) for row in rows)
    require(block.get("label_distribution") == {str(k): v for k, v in sorted(labels.items())}, f"{split}: label distribution drift")
    for row in rows:
        choices = row.get("choice_rows")
        require(isinstance(choices, list) and len(choices) == 4, f"{split}/{row['id']}: expected four choices")
        label = int(row["label"])
        require(0 <= label < 4, f"{split}/{row['id']}: bad label")
        require(bool(row["correct_choice_text_starts_visible"]) == bool(choices[label]["text_starts_visible"]), f"{split}/{row['id']}: correct start drift")
        require(bool(row["correct_choice_text_fully_visible"]) == bool(choices[label]["text_fully_visible"]), f"{split}/{row['id']}: correct full drift")

    return {
        "eligible_rows": eligible_rows,
        "unique_token_sequences": len(set(digests)),
        "duplicate_token_sequence_groups": len(expected_duplicates),
        "no_choice_text_visible_rows": len(no_choice),
        "correct_choice_never_starts_rows": len(correct_missing),
        "all_choices_fully_visible_rows": len(all_full),
        "fractions": fractions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify canonical ARC input-visibility audit evidence.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    require(payload.get("artifact_type") == "LAM-JEPA ARC canonical input visibility audit", "wrong artifact type")
    require(payload.get("test_split_accessed") is False, "ARC test was accessed")
    require(payload.get("canonical_encoder") == "batchify -> text_to_tokens(format_prompt(example), max_len=96)", "encoder contract drift")
    require(payload.get("cutoff_unit") == "whitespace tokens", "cutoff unit drift")
    train = verify_split(payload.get("train") or {}, split="train", source_rows=1119, eligible_rows=1117)
    validation = verify_split(payload.get("validation") or {}, split="validation", source_rows=299, eligible_rows=295)

    report = {
        "verdict": "ARC_CANONICAL_INPUT_VISIBILITY_AUDIT_VERIFIED",
        "test_split_accessed": False,
        "train": train,
        "validation": validation,
        "performance_claim_authorized": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
