from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected object JSON: {path}")
    return payload


def numeric_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    require(bool(ordered), "cannot summarize empty values")

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(len(ordered) - 1, low + 1)
        weight = position - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    return {
        "min": float(ordered[0]),
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def verify_numeric(actual: object, expected: dict[str, float], name: str) -> None:
    require(isinstance(actual, dict), f"{name}: summary missing")
    for key, value in expected.items():
        require(key in actual, f"{name}: missing {key}")
        require(math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12), f"{name}: {key} mismatch")


def verify_split(split: dict, *, expected_source: int, expected_eligible: int, name: str) -> dict[str, object]:
    require(split.get("split") == name, f"{name}: split name mismatch")
    require(split.get("source_rows") == expected_source, f"{name}: source count mismatch")
    require(split.get("eligible_four_choice_rows") == expected_eligible, f"{name}: eligible count mismatch")
    require(split.get("excluded_non_four_choice_rows") == expected_source - expected_eligible, f"{name}: excluded count mismatch")
    require(split.get("fixed_token_length_bytes") == 96, f"{name}: token cutoff drifted")
    require(isinstance(split.get("eligible_id_digest"), str) and len(split["eligible_id_digest"]) == 64, f"{name}: eligible digest invalid")
    require(isinstance(split.get("excluded_id_digest"), str) and len(split["excluded_id_digest"]) == 64, f"{name}: excluded digest invalid")

    rows = split.get("row_records")
    require(isinstance(rows, list) and len(rows) == expected_eligible, f"{name}: row records incomplete")
    ids = [str(row.get("id", "")) for row in rows]
    require(all(ids) and len(ids) == len(set(ids)), f"{name}: duplicate or missing IDs")
    token_digests = [str(row.get("token_digest", "")) for row in rows]
    require(all(len(digest) == 64 for digest in token_digests), f"{name}: invalid token digest")
    require(split.get("unique_token_sequences") == len(set(token_digests)), f"{name}: unique token count mismatch")

    duplicate_expected = [
        {"digest": digest, "count": count}
        for digest, count in Counter(token_digests).items()
        if count > 1
    ]
    require(split.get("duplicate_token_sequence_groups") == duplicate_expected, f"{name}: duplicate-token groups mismatch")

    labels = Counter(int(row["label"]) for row in rows)
    require(split.get("label_distribution") == {str(key): value for key, value in sorted(labels.items())}, f"{name}: label distribution mismatch")

    bool_keys = (
        "choices_marker_visible_before_cutoff",
        "any_choice_text_starts_before_cutoff",
        "all_choice_texts_start_before_cutoff",
        "all_choice_texts_fully_visible_before_cutoff",
        "correct_choice_text_starts_before_cutoff",
        "correct_choice_text_fully_visible_before_cutoff",
    )
    fractions = split.get("fractions")
    require(isinstance(fractions, dict), f"{name}: fractions missing")
    for key in bool_keys:
        expected = sum(bool(row[key]) for row in rows) / len(rows)
        require(math.isclose(float(fractions.get(key, -1.0)), expected, rel_tol=1e-12, abs_tol=1e-12), f"{name}: fraction mismatch {key}")

    verify_numeric(split.get("serialized_bytes"), numeric_summary([float(row["serialized_bytes"]) for row in rows]), f"{name}/serialized_bytes")
    verify_numeric(split.get("retained_fraction"), numeric_summary([float(row["retained_fraction"]) for row in rows]), f"{name}/retained_fraction")
    verify_numeric(split.get("question_bytes"), numeric_summary([float(row["question_bytes"]) for row in rows]), f"{name}/question_bytes")
    verify_numeric(split.get("visible_choice_text_bytes_total"), numeric_summary([float(row["visible_choice_text_bytes_total"]) for row in rows]), f"{name}/visible_choice_text_bytes_total")

    no_choice = [str(row["id"]) for row in rows if not bool(row["any_choice_text_starts_before_cutoff"])]
    correct_missing = [str(row["id"]) for row in rows if not bool(row["correct_choice_text_starts_before_cutoff"])]
    all_full = [str(row["id"]) for row in rows if bool(row["all_choice_texts_fully_visible_before_cutoff"])]
    require(split.get("rows_with_no_choice_text_visible") == no_choice, f"{name}: no-choice row list mismatch")
    require(split.get("rows_where_correct_choice_never_starts") == correct_missing, f"{name}: missing-correct-choice row list mismatch")
    require(split.get("rows_where_all_choices_fully_visible") == all_full, f"{name}: fully-visible row list mismatch")

    for row in rows:
        choices = row.get("choice_rows")
        require(isinstance(choices, list) and len(choices) == 4, f"{name}/{row['id']}: expected four choice audit rows")
        label = int(row["label"])
        require(0 <= label < 4, f"{name}/{row['id']}: invalid label")
        require(bool(row["correct_choice_text_starts_before_cutoff"]) == bool(choices[label]["text_starts_before_cutoff"]), f"{name}/{row['id']}: correct-start flag mismatch")
        require(bool(row["correct_choice_text_fully_visible_before_cutoff"]) == bool(choices[label]["text_fully_visible_before_cutoff"]), f"{name}/{row['id']}: correct-full flag mismatch")
        require(int(row["bytes_retained"]) == min(int(row["serialized_bytes"]), 96), f"{name}/{row['id']}: retained byte count mismatch")

    return {
        "eligible_rows": expected_eligible,
        "unique_token_sequences": len(set(token_digests)),
        "no_choice_text_visible_rows": len(no_choice),
        "correct_choice_never_starts_rows": len(correct_missing),
        "all_choices_fully_visible_rows": len(all_full),
        "fractions": fractions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LAM-JEPA ARC serialization audit evidence.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.audit)
    require(payload.get("artifact_type") == "LAM-JEPA ARC input serialization audit", "artifact type changed")
    require(payload.get("test_split_accessed") is False, "ARC test was accessed")
    require(payload.get("canonical_encoder") == "arc_challenge.encode_example -> text_to_tokens(max_len=96)", "canonical encoder description drifted")
    train = payload.get("train")
    validation = payload.get("validation")
    require(isinstance(train, dict) and isinstance(validation, dict), "split audit missing")

    train_report = verify_split(train, expected_source=1119, expected_eligible=1117, name="train")
    validation_report = verify_split(validation, expected_source=299, expected_eligible=295, name="validation")

    report = {
        "verdict": "ARC_INPUT_SERIALIZATION_AUDIT_VERIFIED",
        "test_split_accessed": False,
        "train": train_report,
        "validation": validation_report,
        "claim_boundary": "This verifies descriptive train/validation serialization evidence only and does not authorize a model-performance claim or confirmatory test access.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
