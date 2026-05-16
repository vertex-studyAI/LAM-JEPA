from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class Misconception:
    name: str
    description: str
    intervention: str
    concept_tags: tuple[str, ...] = ()
    severity: float = 0.5
    modalities: tuple[str, ...] = ("guided_hint", "worked_example")


_DEFAULT_MISCONCEPTIONS = (
    Misconception("sign_error", "Student flips the sign during algebraic manipulation.", "worked_example", ("algebra", "equations", "linear"), 0.8, ("worked_example", "socratic")),
    Misconception("denominator_confusion", "Student mixes up numerator and denominator roles.", "visual_fraction_model", ("fractions", "ratio", "division"), 0.85, ("visual", "analogical")),
    Misconception("unit_mismatch", "Student applies a formula with mismatched units.", "unit_check", ("science", "physics", "measurement"), 0.72, ("visual", "worked_example")),
    Misconception("premise_mismatch", "Student answers from an invalid assumption instead of the prompt.", "prompt_rephrase", ("reading", "reasoning", "comprehension"), 0.7, ("socratic", "counterexample")),
    Misconception("inverse_operation_error", "Student uses the wrong inverse operation when solving.", "hint_chain", ("algebra", "equations"), 0.78, ("guided_hint", "worked_example")),
    Misconception("overgeneralization", "Student transfers a rule to a context where it does not hold.", "counterexample", ("reasoning", "science"), 0.75, ("counterexample", "compare")),
    Misconception("working_memory_overload", "Student likely needs chunking and scaffolding.", "chunked_hint", ("tutoring", "reasoning"), 0.55, ("chunked_hint", "visual")),
    Misconception("retrieval_gap", "Student knows the concept but cannot retrieve it quickly.", "spaced_recall", ("all",), 0.45, ("retrieval_practice", "challenge")),
    Misconception("ratio_confusion", "Student confuses ratio scaling and additive reasoning.", "ratio_table", ("fractions", "ratio"), 0.76, ("visual", "worked_example")),
    Misconception("graph_interpretation", "Student misreads slope, intercept, or axes.", "graph_walkthrough", ("science", "math"), 0.7, ("visual", "socratic")),
    Misconception("causal_fallacy", "Student infers causation from correlation alone.", "counterexample", ("reasoning", "science"), 0.82, ("counterexample", "socratic")),
)


class MisconceptionModel:
    """Rule-and-score hybrid for diagnosing recurring educational misconceptions."""

    def __init__(self, misconceptions: Sequence[Misconception] | None = None):
        self.misconceptions = tuple(misconceptions or _DEFAULT_MISCONCEPTIONS)
        self._counts = {mc.name: 0 for mc in self.misconceptions}
        self._transitions = {mc.name: {other.name: 0 for other in self.misconceptions} for mc in self.misconceptions}

    def _score(self, mc: Misconception, text: str) -> float:
        score = 0.05 + 0.15 * self._counts.get(mc.name, 0)
        for tag in mc.concept_tags:
            if tag in text:
                score += 0.2
        if mc.name == "sign_error" and any(tok in text for tok in ("+", "-", "minus", "equation", "solve")):
            score += 0.35
        if mc.name == "denominator_confusion" and any(tok in text for tok in ("fraction", "ratio", "denominator", "numerator")):
            score += 0.45
        if mc.name == "unit_mismatch" and any(tok in text for tok in ("meter", "newton", "joule", "kelvin", "amp", "volt", "unit")):
            score += 0.4
        if mc.name == "premise_mismatch" and any(tok in text for tok in ("according to", "best evidence", "main idea", "infer")):
            score += 0.35
        if mc.name == "inverse_operation_error" and any(tok in text for tok in ("solve", "isolate", "inverse", "move across")):
            score += 0.4
        if mc.name == "overgeneralization" and any(tok in text for tok in ("always", "every", "all", "never")):
            score += 0.3
        if mc.name == "working_memory_overload" and len(text.split()) > 28:
            score += 0.3
        if mc.name == "retrieval_gap" and any(tok in text for tok in ("remember", "recall", "retrieve")):
            score += 0.25
        if mc.name == "causal_fallacy" and any(tok in text for tok in ("caused", "because of", "correlation", "association")):
            score += 0.4
        return score * (1.0 + mc.severity)

    def rank(self, task: str, prompt: str, answer: str, predicted: str | None = None) -> list[tuple[Misconception, float]]:
        text = f"{task} {prompt} {answer} {predicted or ''}".lower()
        scores = [(mc, self._score(mc, text)) for mc in self.misconceptions]
        scores.sort(key=lambda x: x[1], reverse=True)
        total = sum(max(score, 1e-6) for _, score in scores)
        return [(mc, float(score / total)) for mc, score in scores]

    def diagnose(self, task: str, prompt: str, answer: str, predicted: str | None = None, top_k: int = 3) -> list[dict]:
        ranked = self.rank(task, prompt, answer, predicted=predicted)[:top_k]
        out = []
        for mc, score in ranked:
            out.append({
                "misconception": mc.name,
                "probability": score,
                "description": mc.description,
                "intervention": mc.intervention,
                "modalities": mc.modalities,
                "severity": mc.severity,
            })
        if ranked:
            best = ranked[0][0].name
            self._counts[best] += 1
        return out

    def update_transition(self, previous: str, current: str) -> None:
        if previous in self._transitions and current in self._transitions[previous]:
            self._transitions[previous][current] += 1

    def recommend_intervention(self, misconception_name: str) -> dict:
        for mc in self.misconceptions:
            if mc.name == misconception_name:
                return {"name": mc.intervention, "modalities": mc.modalities, "severity": mc.severity, "description": mc.description}
        return {"name": "guided_hint", "modalities": ("guided_hint",), "severity": 0.4, "description": "generic fallback"}

    def transition_matrix(self) -> torch.Tensor:
        names = [mc.name for mc in self.misconceptions]
        mat = torch.zeros(len(names), len(names), dtype=torch.float32)
        for i, a in enumerate(names):
            row = self._transitions[a]
            total = sum(row.values())
            for j, b in enumerate(names):
                mat[i, j] = float(row[b] / total) if total > 0 else 1.0 / len(names)
        return mat

    def taxonomy(self) -> list[dict]:
        return [
            {
                "name": mc.name,
                "description": mc.description,
                "intervention": mc.intervention,
                "modalities": mc.modalities,
                "severity": mc.severity,
                "concept_tags": mc.concept_tags,
            }
            for mc in self.misconceptions
        ]
