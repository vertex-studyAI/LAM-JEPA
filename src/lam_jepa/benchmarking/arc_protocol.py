from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .arc_challenge import ARCExample, id_digest


ARC_PROTOCOL_CHOICE_COUNT = 4


@dataclass(frozen=True)
class ARCEligibilityResult:
    eligible: tuple[ARCExample, ...]
    excluded: tuple[ARCExample, ...]
    choice_count_distribution: dict[int, int]

    @property
    def original_count(self) -> int:
        return len(self.eligible) + len(self.excluded)

    @property
    def eligible_count(self) -> int:
        return len(self.eligible)

    @property
    def excluded_count(self) -> int:
        return len(self.excluded)

    @property
    def eligible_id_digest(self) -> str:
        return id_digest(self.eligible)

    @property
    def excluded_id_digest(self) -> str:
        return id_digest(self.excluded)


def select_protocol_eligible_examples(
    examples: Sequence[ARCExample],
    *,
    required_choice_count: int = ARC_PROTOCOL_CHOICE_COUNT,
) -> ARCEligibilityResult:
    """Apply the frozen label-independent ARC eligibility rule while preserving source order."""
    if required_choice_count < 2:
        raise ValueError("required_choice_count must be >= 2")

    distribution = Counter(len(example.choices) for example in examples)
    eligible = tuple(example for example in examples if len(example.choices) == required_choice_count)
    excluded = tuple(example for example in examples if len(example.choices) != required_choice_count)

    if len(eligible) + len(excluded) != len(examples):
        raise RuntimeError("ARC eligibility partition does not cover every source row exactly once")
    if {example.item_id for example in eligible} & {example.item_id for example in excluded}:
        raise RuntimeError("ARC eligibility partition contains overlapping item IDs")
    if any(len(example.choices) != required_choice_count for example in eligible):
        raise RuntimeError("ARC eligibility rule admitted a row with the wrong choice count")
    if any(len(example.choices) == required_choice_count for example in excluded):
        raise RuntimeError("ARC eligibility rule excluded an eligible row")

    return ARCEligibilityResult(
        eligible=eligible,
        excluded=excluded,
        choice_count_distribution=dict(sorted(distribution.items())),
    )
