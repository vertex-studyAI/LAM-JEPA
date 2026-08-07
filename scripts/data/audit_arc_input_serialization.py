from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from lam_jepa.benchmarking.arc_challenge import ARCExample, batchify, format_prompt, load_arc_split
from lam_jepa.benchmarking.arc_protocol import select_protocol_eligible_examples
from lam_jepa.data import text_to_tokens

MAX_LEN = 96
VOCAB_SIZE = 256


def token_span(words: list[str], marker: str) -> int:
    try:
        return words.index(marker)
    except ValueError as exc:
        raise RuntimeError(f"missing canonical marker {marker!r}") from exc


def audit_example(example: ARCExample) -> dict[str, object]:
    prompt = format_prompt(example)
    words = prompt.lower().split()
    expected = text_to_tokens(prompt, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    batched, _, _ = batchify([example], vocab_size=VOCAB_SIZE)
    encoded = batched[0].cpu()
    if not torch.equal(encoded, expected):
        raise RuntimeError(f"{example.item_id}: batchify disagrees with text_to_tokens(format_prompt(...))")

    choices_marker = token_span(words, "choices:")
    choice_rows: list[dict[str, object]] = []
    for index, choice in enumerate(example.choices):
        marker = f"[{index}]"
        marker_index = token_span(words, marker)
        next_marker = token_span(words, f"[{index + 1}]") if index + 1 < len(example.choices) else len(words)
        text_start = marker_index + 1
        text_end = next_marker
        choice_rows.append(
            {
                "index": index,
                "marker_token_index": marker_index,
                "text_start_token_index": text_start,
                "text_end_token_index_exclusive": text_end,
                "marker_visible": marker_index < MAX_LEN,
                "text_starts_visible": text_start < MAX_LEN,
                "text_fully_visible": text_end <= MAX_LEN,
                "choice_word_count": max(0, text_end - text_start),
                "visible_choice_words": max(0, min(text_end, MAX_LEN) - min(text_start, MAX_LEN)),
                "choice_text": choice,
            }
        )

    correct = choice_rows[example.label]
    return {
        "id": example.item_id,
        "label": example.label,
        "prompt_word_count": len(words),
        "tokens_retained": min(len(words), MAX_LEN),
        "retained_fraction": min(len(words), MAX_LEN) / max(1, len(words)),
        "question_word_count": len(example.question.lower().split()),
        "choices_marker_token_index": choices_marker,
        "choices_marker_visible": choices_marker < MAX_LEN,
        "any_choice_text_starts_visible": any(bool(row["text_starts_visible"]) for row in choice_rows),
        "all_choice_texts_start_visible": all(bool(row["text_starts_visible"]) for row in choice_rows),
        "all_choice_texts_fully_visible": all(bool(row["text_fully_visible"]) for row in choice_rows),
        "correct_choice_text_starts_visible": bool(correct["text_starts_visible"]),
        "correct_choice_text_fully_visible": bool(correct["text_fully_visible"]),
        "visible_choice_words_total": sum(int(row["visible_choice_words"]) for row in choice_rows),
        "choice_word_count_total": sum(int(row["choice_word_count"]) for row in choice_rows),
        "choice_rows": choice_rows,
        "token_digest": hashlib.sha256(bytes(int(value) for value in encoded.tolist())).hexdigest(),
    }


def numeric(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def summarize(name: str, source: list[ARCExample]) -> dict[str, object]:
    selected = select_protocol_eligible_examples(source)
    rows = [audit_example(example) for example in selected.eligible]
    digests = [str(row["token_digest"]) for row in rows]
    labels = Counter(int(row["label"]) for row in rows)
    bool_keys = (
        "choices_marker_visible",
        "any_choice_text_starts_visible",
        "all_choice_texts_start_visible",
        "all_choice_texts_fully_visible",
        "correct_choice_text_starts_visible",
        "correct_choice_text_fully_visible",
    )
    fractions = {key: sum(bool(row[key]) for row in rows) / len(rows) for key in bool_keys}
    return {
        "split": name,
        "source_rows": selected.original_count,
        "eligible_rows": selected.eligible_count,
        "excluded_rows": selected.excluded_count,
        "eligible_id_digest": selected.eligible_id_digest,
        "excluded_id_digest": selected.excluded_id_digest,
        "max_len_whitespace_tokens": MAX_LEN,
        "unique_token_sequences": len(set(digests)),
        "duplicate_token_sequence_groups": [
            {"digest": digest, "count": count}
            for digest, count in Counter(digests).items()
            if count > 1
        ],
        "label_distribution": {str(key): value for key, value in sorted(labels.items())},
        "prompt_word_count": numeric([float(row["prompt_word_count"]) for row in rows]),
        "retained_fraction": numeric([float(row["retained_fraction"]) for row in rows]),
        "visible_choice_words_total": numeric([float(row["visible_choice_words_total"]) for row in rows]),
        "fractions": fractions,
        "rows_with_no_choice_text_visible": [row["id"] for row in rows if not row["any_choice_text_starts_visible"]],
        "rows_where_correct_choice_never_starts": [row["id"] for row in rows if not row["correct_choice_text_starts_visible"]],
        "rows_where_all_choices_fully_visible": [row["id"] for row in rows if row["all_choice_texts_fully_visible"]],
        "row_records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the actual canonical ARC prompt at the 96-whitespace-token cutoff.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "artifact_type": "LAM-JEPA ARC canonical input visibility audit",
        "test_split_accessed": False,
        "canonical_encoder": "batchify -> text_to_tokens(format_prompt(example), max_len=96)",
        "cutoff_unit": "whitespace tokens",
        "train": summarize("train", load_arc_split(args.train)),
        "validation": summarize("validation", load_arc_split(args.validation)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": payload["train"]["fractions"], "validation": payload["validation"]["fractions"]}, indent=2))


if __name__ == "__main__":
    main()
