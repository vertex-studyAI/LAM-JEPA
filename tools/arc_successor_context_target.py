#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

VIEW_SEED = 20260906
SPAN_FRACTION = 0.20
MAX_SPAN_TOKENS = 8
WITHHELD_MARKER = "[WITHHELD_SPAN]"


@dataclass(frozen=True)
class ContextTargetView:
    example_id: str
    full_supervised_input: str
    jepa_context_input: str
    jepa_target_input: str
    target_start: int
    target_length: int
    question_token_count: int
    withheld_tokens: tuple[str, ...]


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if WITHHELD_MARKER in value:
        raise ValueError(f"{field} must not contain {WITHHELD_MARKER}")
    return value.strip()


def _serialize(question: str, choices: Sequence[str]) -> str:
    parts = [f"Question: {question}"]
    for index, choice in enumerate(choices):
        parts.append(f"Choice {index}: {choice}")
    return "\n".join(parts)


def construct_context_target(
    *,
    example_id: str,
    question_stem: str,
    ordered_choice_texts: Sequence[str],
    view_seed: int = VIEW_SEED,
) -> ContextTargetView:
    """Construct the frozen label-blind successor JEPA view pair.

    The function deliberately has no answer/label argument. Span selection is
    deterministic from the stable example id and the frozen view seed. The
    target is a contiguous span of question-stem whitespace tokens only; choice
    text is preserved in the context and excluded from the target.
    """
    example_id = _require_text(example_id, "example_id")
    question_stem = _require_text(question_stem, "question_stem")
    if not isinstance(ordered_choice_texts, Sequence) or isinstance(ordered_choice_texts, (str, bytes)):
        raise ValueError("ordered_choice_texts must be a non-string sequence")
    if not ordered_choice_texts:
        raise ValueError("ordered_choice_texts must contain at least one choice")
    choices = tuple(_require_text(choice, f"ordered_choice_texts[{index}]") for index, choice in enumerate(ordered_choice_texts))
    if not isinstance(view_seed, int):
        raise ValueError("view_seed must be an integer")

    tokens = question_stem.split()
    if not tokens:
        raise ValueError("question_stem must contain at least one whitespace token")
    span_length = min(len(tokens), max(1, min(MAX_SPAN_TOKENS, math.ceil(SPAN_FRACTION * len(tokens)))))
    digest = hashlib.sha256(f"{view_seed}\0{example_id}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % (len(tokens) - span_length + 1)
    withheld = tuple(tokens[start : start + span_length])
    context_tokens = [*tokens[:start], WITHHELD_MARKER, *tokens[start + span_length :]]
    context_question = " ".join(context_tokens)
    normalized_question = " ".join(tokens)

    return ContextTargetView(
        example_id=example_id,
        full_supervised_input=_serialize(normalized_question, choices),
        jepa_context_input=_serialize(context_question, choices),
        jepa_target_input="Withheld question span: " + " ".join(withheld),
        target_start=start,
        target_length=span_length,
        question_token_count=len(tokens),
        withheld_tokens=withheld,
    )
