#!/usr/bin/env python3
"""Fail-closed verifier for the LAM-JEPA successor budget/collapse gate artifact.

This is a pre-outcome control-plane verifier only. It reads JSON contracts and never
imports model, dataset, training, or scientific runner code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "protocols" / "arc_successor_v1_budget_collapse_gates.json"
PROTOCOL = ROOT / "protocols" / "arc_successor_v1_draft.json"
EXPECTED_PARENT_HEAD = "c62d4bda78380ae9fd76276589b265ced8f48011"
EXPECTED_SEEDS = [11, 23, 37, 53, 71]
EXPECTED_PARENT_BLOCKERS = {
    "DATA_FRESHNESS_AUDIT",
    "CONFIRMATORY_DATASET",
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
    "EXACT_REPRODUCE_COMMAND",
}
EXPECTED_CANDIDATE_CLOSURES = {
    "COLLAPSE_THRESHOLDS",
    "PARAMETER_MATCH_TOLERANCE",
    "MAX_COMPUTE_RATIO",
}


class GateError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _finite_number(value: Any, path: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be numeric")
    numeric = float(value)
    _require(math.isfinite(numeric), f"{path} must be finite")
    return numeric


def verify_gates(gates: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(gates.get("schema_version") == 1, "schema version drift")
    _require(gates.get("artifact_id") == "lam-arc-successor-v1-budget-collapse-gates", "artifact id drift")
    _require(gates.get("status") == "FROZEN_PRE_OUTCOME_NOT_AUTHORIZED", "gate status drift")
    _require(gates.get("stacked_parent_head") == EXPECTED_PARENT_HEAD, "stacked parent head drift")
    _require(gates.get("relationship_to_arc_v5") == "separate_successor_no_rescue_no_reinterpretation", "ARC-v5 relationship drift")
    _require(gates.get("old_locked_test_access") == "PROHIBITED", "old locked test must remain prohibited")
    _require(gates.get("scientific_execution_authorized") is False, "gate artifact must not authorize scientific execution")
    _require(gates.get("outcome_access_authorized") is False, "gate artifact must not authorize outcome access")

    measurement = _mapping(gates.get("measurement_contract"), "measurement_contract")
    _require(measurement.get("probe_surface") == "development_only_fixed_probe_batch", "collapse probe must remain development-only")
    _require(measurement.get("heldout_confirmatory_examples_permitted") is False, "confirmatory examples cannot enter collapse-threshold selection")
    _require(measurement.get("probe_identity_must_be_frozen_before_training") is True, "probe identity must be frozen before training")
    _require(measurement.get("primary_collapse_gate_checkpoint") == "final_training_checkpoint", "primary collapse checkpoint drift")
    _require(measurement.get("rounding_for_decisions") == "none_unrounded_values_only", "decision rounding policy drift")
    _require(measurement.get("nonfinite_value_policy") == "FAIL_CLOSED", "nonfinite policy weakened")
    _require(measurement.get("missing_value_policy") == "FAIL_CLOSED", "missing-value policy weakened")
    _require(measurement.get("failed_or_divergent_seed_policy") == "RETAIN_AND_FAIL_RELEVANT_GATE", "failed-seed policy weakened")

    collapse = _mapping(gates.get("collapse_thresholds"), "collapse_thresholds")
    _require(collapse.get("applies_to_primary_system") == "T1", "primary collapse system drift")
    _require(collapse.get("seed_scope") == EXPECTED_SEEDS, "collapse seed scope drift")
    _require(collapse.get("aggregation") == "every_primary_seed_must_clear_every_gate_at_final_checkpoint", "collapse aggregation drift")
    effective_rank = _mapping(collapse.get("normalized_effective_rank"), "collapse_thresholds.normalized_effective_rank")
    _require(effective_rank.get("definition") == "effective_rank_divided_by_hidden_dimension", "effective-rank definition drift")
    _require(_finite_number(effective_rank.get("minimum_inclusive"), "normalized_effective_rank.minimum_inclusive") == 0.05, "normalized effective-rank threshold drift")
    active = _mapping(collapse.get("active_latent_dimension_fraction"), "collapse_thresholds.active_latent_dimension_fraction")
    _require(_finite_number(active.get("per_dimension_variance_epsilon"), "active_latent_dimension_fraction.per_dimension_variance_epsilon") == 1e-8, "variance epsilon drift")
    _require(_finite_number(active.get("minimum_fraction_inclusive"), "active_latent_dimension_fraction.minimum_fraction_inclusive") == 0.10, "active-dimension threshold drift")
    cosine = _mapping(collapse.get("mean_absolute_pairwise_cosine_similarity"), "collapse_thresholds.mean_absolute_pairwise_cosine_similarity")
    _require(_finite_number(cosine.get("maximum_exclusive"), "mean_absolute_pairwise_cosine_similarity.maximum_exclusive") == 0.995, "cosine-collapse threshold drift")
    norms = _mapping(collapse.get("representation_norms"), "collapse_thresholds.representation_norms")
    _require(norms.get("must_be_finite") is True and norms.get("must_be_strictly_positive") is True, "representation norm gate weakened")
    _require("cannot rescue" in str(collapse.get("secondary_system_boundary", "")).lower(), "secondary-system no-rescue boundary missing")

    params = _mapping(gates.get("parameter_match_tolerance"), "parameter_match_tolerance")
    _require(params.get("primary_pair") == ["B0", "T1"], "parameter-match primary pair drift")
    _require(params.get("must_include_t1_predictor") is True, "T1 predictor must be counted")
    _require(params.get("online_encoder_identity_must_match") is True, "online encoder identity match weakened")
    _require(params.get("classifier_head_shape_must_match") is True, "classifier-head match weakened")
    _require(_finite_number(params.get("maximum_t1_to_b0_gradient_active_parameter_ratio_inclusive"), "parameter ratio") == 1.10, "parameter ratio tolerance drift")
    _require(params.get("missing_parameter_count_policy") == "FAIL_CLOSED", "missing parameter-count policy weakened")
    _require(params.get("post_outcome_architecture_resize_to_meet_ratio") == "PROHIBITED", "post-outcome resize prohibition weakened")
    _require("REPORT_SEPARATELY" in str(params.get("ema_target_parameters", "")), "EMA target accounting boundary missing")

    compute = _mapping(gates.get("compute_budget"), "compute_budget")
    _require(compute.get("primary_pair") == ["B0", "T1"], "compute primary pair drift")
    for key in ("optimizer_steps_ratio", "supervised_label_exposure_ratio", "hyperparameter_trial_count_ratio"):
        _require(_finite_number(compute.get(key), f"compute_budget.{key}") == 1.0, f"{key} must remain exactly matched")
    for key in ("maximum_sequence_length_must_match", "device_class_must_match", "precision_must_match", "must_report_total_training_wall_seconds", "must_report_forward_token_or_example_accounting"):
        _require(compute.get(key) is True, f"compute contract weakened: {key}")
    _require(_finite_number(compute.get("per_seed_maximum_t1_to_b0_accelerator_seconds_ratio_inclusive"), "compute ratio") == 2.0, "compute ratio ceiling drift")
    _require(compute.get("missing_runtime_measurement_policy") == "FAIL_CLOSED", "missing runtime policy weakened")
    _require(compute.get("exceeded_ratio_policy") == "PRIMARY_MECHANISM_CLAIM_FAILS_NO_SECONDARY_RESCUE", "compute overrun policy weakened")

    decision = _mapping(gates.get("decision_integration"), "decision_integration")
    _require(decision.get("secondary_metric_rescue") is False, "secondary metric rescue enabled")
    _require(decision.get("secondary_system_rescue") is False, "secondary system rescue enabled")
    _require(decision.get("posthoc_threshold_movement") is False, "posthoc threshold movement enabled")
    _require(decision.get("posthoc_seed_exclusion") is False, "posthoc seed exclusion enabled")
    required = set(decision.get("primary_positive_result_requires", []))
    _require({"frozen_H1_accuracy_and_uncertainty_gate", "frozen_H2_seed_consistency_gate", "this_artifact_H3_noncollapse_gate", "this_artifact_parameter_match_gate", "this_artifact_compute_budget_gate"} == required, "primary positive-result conjunction drift")

    boundary = _mapping(gates.get("stacked_resolution_boundary"), "stacked_resolution_boundary")
    _require(set(boundary.get("candidate_control_definitions_closed_here", [])) == EXPECTED_CANDIDATE_CLOSURES, "candidate closure set drift")
    _require(boundary.get("canonical_parent_protocol_blocker_ledger_intentionally_unchanged") is True, "stacked parent-ledger boundary weakened")
    _require(set(boundary.get("parent_current_hard_blockers_must_still_include", [])) == EXPECTED_PARENT_BLOCKERS, "parent blocker snapshot drift")
    _require(boundary.get("execution_remains_blocked") is True, "execution boundary weakened")

    _require(protocol.get("status") == "DRAFT_NOT_FROZEN", "parent protocol must remain draft in this stacked lane")
    _require(protocol.get("execution_authorized") is False, "parent protocol must remain unauthorized")
    _require(protocol.get("old_locked_test_access") == "PROHIBITED", "parent old locked test policy drift")
    _require(protocol.get("proposed_seeds") == EXPECTED_SEEDS, "parent seed list drift")
    parent_blockers = protocol.get("hard_blockers")
    _require(isinstance(parent_blockers, list) and EXPECTED_PARENT_BLOCKERS.issubset(set(parent_blockers)), "parent hard blocker removed concurrently")

    return {
        "status": "ARC_SUCCESSOR_BUDGET_COLLAPSE_GATES_VERIFIED_PREOUTCOME",
        "scientific_execution_authorized": False,
        "outcome_access_authorized": False,
        "old_locked_test_access": "PROHIBITED",
        "candidate_control_definitions_closed": sorted(EXPECTED_CANDIDATE_CLOSURES),
        "canonical_parent_blocker_ledger_unchanged": True,
        "parent_hard_blockers_retained": sorted(EXPECTED_PARENT_BLOCKERS),
        "frozen_seeds": EXPECTED_SEEDS,
        "collapse_thresholds": {
            "normalized_effective_rank_min": 0.05,
            "active_dimension_fraction_min": 0.10,
            "variance_epsilon": 1e-8,
            "mean_absolute_pairwise_cosine_max_exclusive": 0.995,
        },
        "parameter_ratio_max": 1.10,
        "accelerator_seconds_ratio_max_per_seed": 2.0,
    }


def verify_repository() -> dict[str, Any]:
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    result = verify_gates(gates, protocol)
    result["gates_sha256"] = hashlib.sha256(GATES.read_bytes()).hexdigest()
    result["parent_protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = verify_repository()
    except (GateError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "FAIL_ARC_SUCCESSOR_BUDGET_COLLAPSE_GATES", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
