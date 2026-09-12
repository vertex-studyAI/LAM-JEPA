from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from arc_successor_context_target import (  # noqa: E402
    MAX_SPAN_TOKENS,
    SPAN_FRACTION,
    VIEW_SEED,
    WITHHELD_MARKER,
    construct_context_target,
)
from verify_arc_successor_context_target import (  # noqa: E402
    ContextTargetFreezeError,
    verify,
)

ARTIFACT = ROOT / "protocols" / "arc_successor_v1_context_target.json"
SUCCESSOR = ROOT / "protocols" / "arc_successor_v1_draft.json"
IMPLEMENTATION = ROOT / "tools" / "arc_successor_context_target.py"


class ArcSuccessorContextTargetTests(unittest.TestCase):
    def test_repository_freeze_verifies_without_authorizing_execution(self) -> None:
        result = verify(ARTIFACT, SUCCESSOR, IMPLEMENTATION)
        self.assertEqual(result["status"], "ARC_SUCCESSOR_CONTEXT_TARGET_FROZEN_VERIFIED")
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["outcome_access_authorized"])
        self.assertEqual(len(result["remaining_hard_blockers"]), 6)
        self.assertNotIn("ENCODER_FAMILY_AND_REVISION", result["remaining_hard_blockers"])

    def test_constructor_signature_has_no_label_surface(self) -> None:
        signature = inspect.signature(construct_context_target)
        self.assertEqual(
            tuple(signature.parameters),
            ("example_id", "question_stem", "ordered_choice_texts", "view_seed"),
        )
        for forbidden in ("label", "answer_key", "gold", "correct_choice_index"):
            self.assertNotIn(forbidden, signature.parameters)

    def test_view_is_deterministic_and_uses_frozen_span_rule(self) -> None:
        question = "one two three four five six seven eight nine ten eleven twelve"
        choices = ("red-only-choice", "blue-only-choice", "green-only-choice", "yellow-only-choice")
        first = construct_context_target(
            example_id="fixture-17",
            question_stem=question,
            ordered_choice_texts=choices,
        )
        second = construct_context_target(
            example_id="fixture-17",
            question_stem=question,
            ordered_choice_texts=choices,
        )
        self.assertEqual(first, second)
        expected_length = min(12, max(1, min(MAX_SPAN_TOKENS, math.ceil(SPAN_FRACTION * 12))))
        self.assertEqual(first.target_length, expected_length)
        self.assertEqual(VIEW_SEED, 20260906)
        self.assertEqual(first.question_token_count, 12)
        self.assertIn(WITHHELD_MARKER, first.jepa_context_input)
        self.assertNotEqual(first.full_supervised_input, first.jepa_context_input)

    def test_choices_remain_context_only_and_order_is_preserved(self) -> None:
        choices = ("choice-aaa", "choice-bbb", "choice-ccc", "choice-ddd")
        view = construct_context_target(
            example_id="fixture-choice-order",
            question_stem="which synthetic property is selected from this deliberately neutral question",
            ordered_choice_texts=choices,
        )
        positions = [view.jepa_context_input.index(choice) for choice in choices]
        self.assertEqual(positions, sorted(positions))
        for choice in choices:
            self.assertIn(choice, view.full_supervised_input)
            self.assertIn(choice, view.jepa_context_input)
            self.assertNotIn(choice, view.jepa_target_input)

    def test_withheld_tokens_are_exact_contiguous_question_slice(self) -> None:
        tokens = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
        view = construct_context_target(
            example_id="fixture-contiguous",
            question_stem=" ".join(tokens),
            ordered_choice_texts=("c0", "c1", "c2", "c3"),
        )
        expected = tuple(tokens[view.target_start : view.target_start + view.target_length])
        self.assertEqual(view.withheld_tokens, expected)
        self.assertEqual(view.jepa_target_input, "Withheld question span: " + " ".join(expected))

    def test_invalid_or_ambiguous_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "example_id"):
            construct_context_target(example_id="", question_stem="valid question", ordered_choice_texts=("a", "b"))
        with self.assertRaisesRegex(ValueError, "question_stem"):
            construct_context_target(example_id="id", question_stem="", ordered_choice_texts=("a", "b"))
        with self.assertRaisesRegex(ValueError, "at least one choice"):
            construct_context_target(example_id="id", question_stem="valid question", ordered_choice_texts=())
        with self.assertRaisesRegex(ValueError, "WITHHELD_SPAN"):
            construct_context_target(
                example_id="id",
                question_stem="source contains [WITHHELD_SPAN] marker",
                ordered_choice_texts=("a", "b"),
            )

    def test_artifact_cannot_self_authorize_or_relax_label_boundary(self) -> None:
        payload = json.loads(ARTIFACT.read_text())
        successor = json.loads(SUCCESSOR.read_text())
        for field in ("execution_authorized", "outcome_access_authorized"):
            changed = copy.deepcopy(payload)
            changed[field] = True
            temp = ROOT / "tests" / f".tmp-context-{field}.json"
            try:
                temp.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ContextTargetFreezeError):
                    verify(temp, SUCCESSOR, IMPLEMENTATION)
            finally:
                temp.unlink(missing_ok=True)

        self.assertFalse(payload["input_contract"]["construction_may_read_supervised_label"])
        self.assertFalse(payload["visibility_rules"]["target_receives_answer_or_label"])
        self.assertNotIn("CONTEXT_TARGET_CONSTRUCTION", successor["hard_blockers"])
        self.assertNotIn("ENCODER_FAMILY_AND_REVISION", successor["hard_blockers"])


if __name__ == "__main__":
    unittest.main()
