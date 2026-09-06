#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

EXPECTED_ARTIFACT_SHA256 = "6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b"
EXPECTED_BLOCKERS_AFTER_FREEZE = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "ENCODER_FAMILY_AND_REVISION",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}


class ContextTargetFreezeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContextTargetFreezeError(message)


def as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return as_mapping(payload, str(path))


def load_construction_module(path: Path):
    spec = importlib.util.spec_from_file_location("arc_successor_context_target", path)
    require(spec is not None and spec.loader is not None, "cannot load context-target implementation")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect sys.modules while class decorators execute. Register the
    # synthetic module before exec_module so exact-head CI exercises the same
    # import semantics as a normal Python import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(artifact_path: Path, successor_path: Path, implementation_path: Path) -> dict[str, Any]:
    artifact_bytes = artifact_path.read_bytes()
    artifact = load_json(artifact_path)
    successor = load_json(successor_path)
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    require(digest == EXPECTED_ARTIFACT_SHA256, "context-target artifact byte identity drift")

    require(artifact.get("schema_version") == 1, "schema_version drift")
    require(artifact.get("artifact_id") == "lam-jepa-successor-context-target-v1-20260906", "artifact_id drift")
    require(artifact.get("status") == "FROZEN_PREOUTCOME_CONSTRUCTION_ONLY", "status drift")
    require(artifact.get("execution_authorized") is False, "context-target freeze must not authorize execution")
    require(artifact.get("outcome_access_authorized") is False, "context-target freeze must not authorize outcome access")

    input_contract = as_mapping(artifact.get("input_contract"), "input_contract")
    require(input_contract.get("required_fields") == ["example_id", "question_stem", "ordered_choice_texts"], "required input contract drift")
    forbidden = set(input_contract.get("forbidden_fields") or [])
    require({"answer_key", "correct_choice_index", "label", "gold", "target_label"}.issubset(forbidden), "label-bearing fields must remain forbidden")
    require(input_contract.get("construction_may_read_supervised_label") is False, "view construction must remain label-blind")

    selection = as_mapping(artifact.get("target_selection"), "target_selection")
    require(selection.get("source_region") == "question_stem_only", "target source region drift")
    require(selection.get("tokenization_for_span_selection") == "unicode_whitespace_split", "span tokenization drift")
    require(selection.get("view_seed") == 20260906, "view seed drift")
    require(float(selection.get("span_fraction", -1)) == 0.20, "span fraction drift")
    require(selection.get("one_view_per_example") is True, "one-view contract drift")
    require(selection.get("changes_across_epochs") is False, "view must not change across epochs")
    require(selection.get("changes_across_model_seeds") is False, "view must not change across model seeds")

    visibility = as_mapping(artifact.get("visibility_rules"), "visibility_rules")
    for key in (
        "context_receives_withheld_tokens",
        "context_receives_answer_or_label",
        "target_receives_context_remainder",
        "target_receives_choice_texts",
        "target_receives_answer_or_label",
    ):
        require(visibility.get(key) is False, f"{key} must remain false")
    require("gold label is used only by supervised loss" in str(visibility.get("supervised_classifier_input", "")), "supervised label boundary drift")

    pairing = as_mapping(artifact.get("pairing_contract"), "pairing_contract")
    require(pairing.get("classifier_architecture_or_label_exposure_changed_by_this_artifact") is False, "classifier/label pairing boundary drift")
    require(pairing.get("T1_supervised_input") == "same full label-free serialization as B0", "B0/T1 supervised pairing drift")

    integrity = as_mapping(artifact.get("integrity_rules"), "integrity_rules")
    require(all(value is False for value in integrity.values()), "post-outcome/manual view changes must remain forbidden")
    boundary = str(artifact.get("claim_boundary", "")).lower()
    require("not evidence" in boundary and "authorized for confirmatory execution" in boundary, "claim boundary drift")

    require(successor.get("status") == "DRAFT_NOT_FROZEN", "successor must remain draft")
    require(successor.get("execution_authorized") is False, "successor must remain non-authorized")
    hard_blockers = set(successor.get("hard_blockers") or [])
    require(hard_blockers == EXPECTED_BLOCKERS_AFTER_FREEZE, "successor hard-blocker set drift")
    require("CONTEXT_TARGET_CONSTRUCTION" not in hard_blockers, "resolved context-target blocker must not remain hard-blocked")
    resolved = as_mapping(successor.get("resolved_blockers"), "resolved_blockers")
    binding = as_mapping(resolved.get("CONTEXT_TARGET_CONSTRUCTION"), "resolved_blockers.CONTEXT_TARGET_CONSTRUCTION")
    require(binding.get("artifact") == "protocols/arc_successor_v1_context_target.json", "context-target artifact binding drift")
    require(binding.get("sha256") == digest, "context-target SHA-256 binding drift")

    module = load_construction_module(implementation_path)
    signature = inspect.signature(module.construct_context_target)
    require(tuple(signature.parameters) == ("example_id", "question_stem", "ordered_choice_texts", "view_seed"), "implementation signature drift")
    require(module.VIEW_SEED == 20260906, "implementation view seed drift")
    require(module.SPAN_FRACTION == 0.20, "implementation span fraction drift")
    require(module.MAX_SPAN_TOKENS == 8, "implementation max span drift")
    require(module.WITHHELD_MARKER == "[WITHHELD_SPAN]", "implementation marker drift")

    question = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    choices = ("choice-red", "choice-blue", "choice-green", "choice-yellow")
    view = module.construct_context_target(example_id="synthetic-fixture-001", question_stem=question, ordered_choice_texts=choices)
    expected_length = min(10, max(1, min(8, math.ceil(0.20 * 10))))
    require(view.target_length == expected_length, "implementation span length disagrees with frozen rule")
    require(module.WITHHELD_MARKER in view.jepa_context_input, "implementation context lacks withheld marker")
    require(all(choice in view.jepa_context_input for choice in choices), "implementation must preserve ordered choice text in context")
    require(all(choice not in view.jepa_target_input for choice in choices), "implementation target must exclude choices")
    repeat = module.construct_context_target(example_id="synthetic-fixture-001", question_stem=question, ordered_choice_texts=choices)
    require(view == repeat, "implementation must be deterministic")

    return {
        "status": "ARC_SUCCESSOR_CONTEXT_TARGET_FROZEN_VERIFIED",
        "artifact_sha256": digest,
        "execution_authorized": False,
        "outcome_access_authorized": False,
        "remaining_hard_blockers": sorted(hard_blockers),
        "synthetic_target_start": view.target_start,
        "synthetic_target_length": view.target_length,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("protocols/arc_successor_v1_context_target.json"))
    parser.add_argument("--successor", type=Path, default=Path("protocols/arc_successor_v1_draft.json"))
    parser.add_argument("--implementation", type=Path, default=Path("tools/arc_successor_context_target.py"))
    args = parser.parse_args()
    try:
        result = verify(args.artifact, args.successor, args.implementation)
    except (OSError, json.JSONDecodeError, ContextTargetFreezeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
