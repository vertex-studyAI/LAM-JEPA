from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from lam_jepa.benchmarking.arc_challenge import ARCExample, encode_example, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.data import text_to_tokens

MAX_LEN = 96
VOCAB_SIZE = 256


def serialize_example(example: ARCExample) -> str:
    return "Question: " + example.question + "\nChoices:\n" + "\n".join(
        f"{index}: {choice}" for index, choice in enumerate(example.choices)
    )


def byte_span(haystack: bytes, needle: bytes) -> tuple[int, int] | None:
    start = haystack.find(needle)
    if start < 0:
        return None
    return start, start + len(needle)


def audit_example(example: ARCExample) -> dict[str, object]:
    serialized = serialize_example(example)
    payload = serialized.encode("utf-8", errors="ignore")
    encoded = encode_example(example, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    expected = text_to_tokens(serialized, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    if not torch.equal(encoded, expected):
        raise RuntimeError(f"{example.item_id}: audit serialization drifted from canonical encode_example")

    choices_marker = b"\nChoices:\n"
    choices_span = byte_span(payload, choices_marker)
    if choices_span is None:
        raise RuntimeError(f"{example.item_id}: serialized Choices marker missing")

    choice_rows: list[dict[str, object]] = []
    for index, choice in enumerate(example.choices):
        marker = f"\n{index}: ".encode("utf-8") if index > 0 else f"{index}: ".encode("utf-8")
        search_from = choices_span[1]
        marker_start = payload.find(marker, search_from)
        if marker_start < 0:
            raise RuntimeError(f"{example.item_id}: choice marker {index} missing")
        text_start = marker_start + len(marker)
        choice_bytes = choice.encode("utf-8", errors="ignore")
        text_end = text_start + len(choice_bytes)
        choice_rows.append(
            {
                "index": index,
                "marker_start_byte": marker_start,
                "text_start_byte": text_start,
                "text_end_byte": text_end,
                "marker_visible_before_cutoff": marker_start < MAX_LEN,
                "text_starts_before_cutoff": text_start < MAX_LEN,
                "text_fully_visible_before_cutoff": text_end <= MAX_LEN,
                "visible_text_bytes": max(0, min(text_end, MAX_LEN) - min(text_start, MAX_LEN)),
                "choice_text_bytes": len(choice_bytes),
            }
        )

    correct = choice_rows[example.label]
    return {
        "id": example.item_id,
        "label": example.label,
        "serialized_bytes": len(payload),
        "bytes_retained": min(len(payload), MAX_LEN),
        "retained_fraction": min(len(payload), MAX_LEN) / max(1, len(payload)),
        "question_bytes": len(example.question.encode("utf-8", errors="ignore")),
        "choices_marker_start_byte": choices_span[0],
        "choices_marker_visible_before_cutoff": choices_span[0] < MAX_LEN,
        "any_choice_text_starts_before_cutoff": any(bool(row["text_starts_before_cutoff"]) for row in choice_rows),
        "all_choice_texts_start_before_cutoff": all(bool(row["text_starts_before_cutoff"]) for row in choice_rows),
        "all_choice_texts_fully_visible_before_cutoff": all(bool(row["text_fully_visible_before_cutoff"]) for row in choice_rows),
        "correct_choice_text_starts_before_cutoff": bool(correct["text_starts_before_cutoff"]),
        "correct_choice_text_fully_visible_before_cutoff": bool(correct["text_fully_visible_before_cutoff"]),
        "visible_choice_text_bytes_total": sum(int(row["visible_text_bytes"]) for row in choice_rows),
        "choice_text_bytes_total": sum(int(row["choice_text_bytes"]) for row in choice_rows),
        "choice_rows": choice_rows,
        "token_digest": hashlib.sha256(bytes(int(value) for value in encoded.tolist())).hexdigest(),
    }


def fraction(rows: list[dict[str, object]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def numeric_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
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


def summarize_split(name: str, source: list[ARCExample]) -> dict[str, object]:
    partition = select_protocol_eligible_examples(source)
    examples = list(partition.eligible)
    rows = [audit_example(example) for example in examples]
    token_digests = [str(row["token_digest"]) for row in rows]
    duplicate_token_groups = [
        {"digest": digest, "count": count}
        for digest, count in Counter(token_digests).items()
        if count > 1
    ]
    labels = Counter(example.label for example in examples)

    summary = {
        "split": name,
        "source_rows": partition.original_count,
        "eligible_four_choice_rows": partition.eligible_count,
        "excluded_non_four_choice_rows": partition.excluded_count,
        "eligible_id_digest": partition.eligible_id_digest,
        "excluded_id_digest": partition.excluded_id_digest,
        "fixed_token_length_bytes": MAX_LEN,
        "unique_token_sequences": len(set(token_digests)),
        "duplicate_token_sequence_groups": duplicate_token_groups,
        "label_distribution": {str(key): value for key, value in sorted(labels.items())},
        "serialized_bytes": numeric_summary([float(row["serialized_bytes"]) for row in rows]),
        "retained_fraction": numeric_summary([float(row["retained_fraction"]) for row in rows]),
        "question_bytes": numeric_summary([float(row["question_bytes"]) for row in rows]),
        "visible_choice_text_bytes_total": numeric_summary([float(row["visible_choice_text_bytes_total"]) for row in rows]),
        "fractions": {
            "choices_marker_visible_before_cutoff": fraction(rows, "choices_marker_visible_before_cutoff"),
            "any_choice_text_starts_before_cutoff": fraction(rows, "any_choice_text_starts_before_cutoff"),
            "all_choice_texts_start_before_cutoff": fraction(rows, "all_choice_texts_start_before_cutoff"),
            "all_choice_texts_fully_visible_before_cutoff": fraction(rows, "all_choice_texts_fully_visible_before_cutoff"),
            "correct_choice_text_starts_before_cutoff": fraction(rows, "correct_choice_text_starts_before_cutoff"),
            "correct_choice_text_fully_visible_before_cutoff": fraction(rows, "correct_choice_text_fully_visible_before_cutoff"),
        },
        "rows_with_no_choice_text_visible": [row["id"] for row in rows if not row["any_choice_text_starts_before_cutoff"]],
        "rows_where_correct_choice_never_starts": [row["id"] for row in rows if not row["correct_choice_text_starts_before_cutoff"]],
        "rows_where_all_choices_fully_visible": [row["id"] for row in rows if row["all_choice_texts_fully_visible_before_cutoff"]],
        "row_records": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical ARC serialization before confirmatory test access.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train = load_arc_split(args.train)
    validation = load_arc_split(args.validation)
    payload = {
        "artifact_type": "LAM-JEPA ARC input serialization audit",
        "test_split_accessed": False,
        "canonical_encoder": "arc_challenge.encode_example -> text_to_tokens(max_len=96)",
        "train": summarize_split("train", train),
        "validation": summarize_split("validation", validation),
        "interpretation_rule": (
            "This audit is descriptive only. If a large fraction of eligible rows do not expose candidate answer text within the fixed 96-byte input, "
            "then the current ARC adapter cannot support a clean answer-reasoning claim without a new pre-test protocol/implementation version."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": payload["train"]["fractions"], "validation": payload["validation"]["fractions"]}, indent=2))


if __name__ == "__main__":
    main()
