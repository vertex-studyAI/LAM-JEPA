from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class DecisionRuleError(ValueError):
    pass


DECISION_RESOLVED = {"DELTA_PRIMARY", "SEED_WIN_FRACTION", "UNCERTAINTY_RULE"}
UNRESOLVED_AT_DECISION_FREEZE = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "ENCODER_FAMILY_AND_REVISION",
    "CONTEXT_TARGET_CONSTRUCTION",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}
SUPPORTED_LATER_RESOLUTIONS = {
    "CONTEXT_TARGET_CONSTRUCTION": {
        "artifact": "protocols/arc_successor_v1_context_target.json",
        "sha256": "6d828536b9983f84beb85e29c1db56dfd01a23aa22f3840a88272867bd3f5d6b",
    },
    "ENCODER_FAMILY_AND_REVISION": {
        "artifact": "protocols/arc_successor_v1_encoder.json",
        "sha256": "8d9ea79aa7ee9777d3aad9044b63f1db5813b6e214d2bc7d559539700f09b516",
        "repo_id": "distilbert/distilroberta-base",
        "revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b",
        "checkpoint_sha256": "2b11ca9cf3d2cbb44cc1a93ad96aedc2894231ae6e33e2d2268c3d7b1ff97663",
        "local_byte_receipts_retained": False,
    },
}
SEEDS = [11, 23, 37, 53, 71]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionRuleError(message)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def verify_rules(rules: Mapping[str, Any], successor: Mapping[str, Any]) -> dict[str, Any]:
    _require(rules.get("schema_version") == 1, "schema_version must equal 1")
    _require(
        rules.get("decision_rules_id")
        == "lam-arc-contextual-successor-v1-preoutcome-decision-rules-20260906",
        "decision_rules_id drift",
    )
    _require(
        rules.get("protocol_id") == successor.get("protocol_id") == "lam-arc-contextual-successor-v1-draft",
        "protocol_id mismatch",
    )
    _require(rules.get("status") == "PREOUTCOME_FROZEN_DECISION_RULES", "decision rules must remain pre-outcome frozen")
    _require(rules.get("execution_authorized") is False, "decision rules must not authorize execution")
    _require(rules.get("outcomes_observed") is False, "decision rules must remain pre-outcome")

    _require(successor.get("status") == "DRAFT_NOT_FROZEN", "successor must remain DRAFT_NOT_FROZEN")
    _require(successor.get("execution_authorized") is False, "successor must remain execution_authorized=false")

    metric = rules.get("primary_metric")
    _require(metric == "held_out_multiple_choice_accuracy", "primary metric drift")

    comparison = _as_mapping(rules.get("comparison"), "comparison")
    _require(comparison.get("treatment") == "T1", "primary treatment must remain T1")
    _require(comparison.get("baseline") == "B0", "primary baseline must remain B0")
    _require("same confirmatory examples" in str(comparison.get("pairing", "")), "pairing contract drift")

    material = _as_mapping(rules.get("material_effect"), "material_effect")
    _require(float(material.get("delta_primary_absolute_accuracy", -1)) == 0.02, "DELTA_PRIMARY must equal 0.02")
    _require(
        material.get("h1_gate") == "mean_paired_accuracy_delta >= 0.02 AND bootstrap_95pct_lower_bound > 0",
        "H1 gate drift",
    )

    seed = _as_mapping(rules.get("seed_consistency"), "seed_consistency")
    _require(seed.get("frozen_seeds") == SEEDS, "frozen seed list drift")
    _require(seed.get("minimum_positive_seed_count") == 4, "minimum positive seed count must equal 4")
    _require(float(seed.get("seed_win_fraction", -1)) == 0.8, "SEED_WIN_FRACTION must equal 0.8")
    _require(seed.get("positive_definition") == "per_seed_paired_accuracy_delta > 0", "positive seed definition drift")
    _require(seed.get("ties_are_wins") is False, "ties must not count as wins")
    _require(seed.get("failed_or_collapsed_seeds_may_not_be_dropped") is True, "failed/collapsed seeds must be retained")

    uncertainty = _as_mapping(rules.get("uncertainty"), "uncertainty")
    expected = {
        "method": "paired_hierarchical_bootstrap",
        "confidence_level": 0.95,
        "interval": "percentile",
        "replicates": 10000,
        "bootstrap_seed": 20260906,
        "outer_resampling_unit": "frozen_seed",
        "inner_resampling_unit": "confirmatory_example_within_selected_seed",
        "pairing_preserved": True,
        "directional_gate": "lower endpoint must be strictly greater than 0",
        "rounding_policy": "evaluate gates on unrounded floating-point values; round only for display",
    }
    for key, value in expected.items():
        _require(uncertainty.get(key) == value, f"uncertainty.{key} drift")
    _require("1[T1 correct]-1[B0 correct]" in str(uncertainty.get("inner_pair_definition", "")), "paired correctness definition drift")

    logic = _as_mapping(rules.get("decision_logic"), "decision_logic")
    _require(
        logic.get("primary_success_requires_all") == ["H1_material_effect", "H2_seed_consistency", "H3_noncollapse"],
        "primary success conjunction drift",
    )
    _require(logic.get("secondary_metrics_can_rescue_primary_failure") is False, "secondary rescue must remain forbidden")
    _require(logic.get("posthoc_seed_exclusion") is False, "posthoc seed exclusion must remain forbidden")
    _require(logic.get("posthoc_threshold_change") is False, "posthoc threshold changes must remain forbidden")
    _require(logic.get("null_or_negative_is_valid_terminal_outcome") is True, "null/negative terminal outcome must remain valid")

    resolved_at_freeze = set(rules.get("resolved_successor_blockers") or [])
    unresolved_at_freeze = set(rules.get("still_unresolved_successor_blockers") or [])
    _require(resolved_at_freeze == DECISION_RESOLVED, "resolved blocker set drift in immutable decision artifact")
    _require(unresolved_at_freeze == UNRESOLVED_AT_DECISION_FREEZE, "remaining blocker set drift in immutable decision artifact")

    resolved_map = _as_mapping(successor.get("resolved_blockers"), "successor.resolved_blockers")
    resolved_now = set(resolved_map)
    _require(DECISION_RESOLVED.issubset(resolved_now), "successor lost frozen decision-rule resolutions")
    later_resolved = resolved_now - DECISION_RESOLVED
    _require(later_resolved.issubset(SUPPORTED_LATER_RESOLUTIONS), "unsupported later successor blocker resolution")
    expected_hard_blockers = UNRESOLVED_AT_DECISION_FREEZE - later_resolved
    successor_blockers = set(successor.get("hard_blockers") or [])
    _require(successor_blockers == expected_hard_blockers, "successor hard_blockers do not match independently bound later resolutions")

    for blocker in sorted(DECISION_RESOLVED):
        binding = _as_mapping(resolved_map[blocker], f"successor.resolved_blockers.{blocker}")
        _require(binding.get("artifact") == "protocols/arc_successor_v1_decision_rules.json", f"{blocker} artifact binding drift")
        digest = str(binding.get("sha256", ""))
        _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"{blocker} sha256 must be lowercase hex")

    for blocker in sorted(later_resolved):
        binding = _as_mapping(resolved_map[blocker], f"successor.resolved_blockers.{blocker}")
        expected_binding = SUPPORTED_LATER_RESOLUTIONS[blocker]
        _require(dict(binding) == expected_binding, f"{blocker} later-resolution binding drift")

    _require("do not authorize execution" in str(rules.get("claim_boundary", "")).lower(), "claim boundary must preserve non-authorization")

    return {
        "status": "ARC_SUCCESSOR_PREOUTCOME_DECISION_RULES_VERIFIED",
        "decision_rule_resolved_blockers": sorted(DECISION_RESOLVED),
        "later_resolved_blockers": sorted(later_resolved),
        "remaining_blockers": sorted(successor_blockers),
        "primary_metric": metric,
        "delta_primary": 0.02,
        "seed_win_fraction": 0.8,
        "bootstrap_replicates": 10000,
        "execution_authorized": False,
    }


def _load(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text())
    _require(isinstance(data, Mapping), f"{path} must contain a JSON object")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rules", type=Path)
    parser.add_argument("successor", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_rules(_load(args.rules), _load(args.successor))
    except (OSError, json.JSONDecodeError, DecisionRuleError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
