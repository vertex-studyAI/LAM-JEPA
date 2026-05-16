from __future__ import annotations

from dataclasses import dataclass

from .misconception_model import Misconception


@dataclass
class Intervention:
    name: str
    explanation_type: str
    prompt_style: str
    confidence_level: float
    rationale: str
    modality: str = "text"
    estimated_gain: float = 0.5


class InterventionSelector:
    """Selects the best tutoring move using misconception, mastery and confidence evidence."""

    def select(self, misconception: str | Misconception | None, confidence: float, mastery: float, difficulty: float, fatigue: float = 0.0, retention: float = 0.5) -> Intervention:
        name = misconception.name if isinstance(misconception, Misconception) else (misconception or "generic_hint")
        low_conf = confidence < 0.45
        low_mastery = mastery < 0.4
        high_fatigue = fatigue > 0.65
        low_retention = retention < 0.45

        if name in {"denominator_confusion", "unit_mismatch", "graph_interpretation"}:
            explanation_type = "visual"
            prompt_style = "diagram-first"
            modality = "visual"
            rationale = "visual scaffolds reduce representational confusion"
            gain = 0.8
        elif name in {"sign_error", "inverse_operation_error", "ratio_confusion"}:
            explanation_type = "worked_example"
            prompt_style = "step-by-step"
            modality = "symbolic"
            rationale = "a concrete algebraic trace exposes the transformation rule"
            gain = 0.78
        elif name in {"premise_mismatch", "overgeneralization", "causal_fallacy"}:
            explanation_type = "counterexample"
            prompt_style = "compare-and-contrast"
            modality = "text"
            rationale = "contrasting examples reveal the hidden assumption"
            gain = 0.82
        elif low_conf and low_mastery:
            explanation_type = "chunked_hint"
            prompt_style = "micro-steps"
            modality = "text"
            rationale = "the learner likely needs a smaller reasoning unit"
            gain = 0.76
        elif high_fatigue or low_retention:
            explanation_type = "retrieval_practice"
            prompt_style = "spaced-recall"
            modality = "text"
            rationale = "the learner benefits from a lighter recall prompt rather than new content"
            gain = 0.7
        elif mastery > 0.75 and difficulty > 0.6:
            explanation_type = "challenge_problem"
            prompt_style = "transfer"
            modality = "challenge"
            rationale = "the learner is ready for an extension task"
            gain = 0.9
        else:
            explanation_type = "guided_hint"
            prompt_style = "targeted"
            modality = "text"
            rationale = "the learner benefits from a precise nudge"
            gain = 0.74

        if confidence > 0.8 and mastery > 0.8:
            gain += 0.05
        if low_conf:
            gain -= 0.05
        if difficulty > 0.75 and low_mastery:
            gain += 0.03

        return Intervention(
            name=name,
            explanation_type=explanation_type,
            prompt_style=prompt_style,
            confidence_level=float(max(0.0, min(1.0, confidence))),
            rationale=rationale,
            modality=modality,
            estimated_gain=float(max(0.0, min(1.0, gain))),
        )
