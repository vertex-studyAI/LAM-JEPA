from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

CHECKPOINTS = [0, 1, 5, 10, 25, 50, 100, 200, 300]
THRESHOLD = 0.95


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def summarize(records: list[dict]) -> dict[str, object]:
    finals = [float(record["history"][-1]["accuracy"]) for record in records]
    bests = [max(float(point["accuracy"]) for point in record["history"]) for record in records]
    return {
        "final_accuracy_by_seed": finals,
        "best_accuracy_by_seed": bests,
        "mean_final_accuracy": float(statistics.fmean(finals)),
        "mean_best_accuracy": float(statistics.fmean(bests)),
        "all_seeds_reach_overfit_threshold": all(value >= THRESHOLD for value in bests),
    }


def classify(summaries: dict[str, dict[str, object]]) -> str:
    matched = bool(summaries["matched_supervised"]["all_seeds_reach_overfit_threshold"])
    ce_only = bool(summaries["lam_ce_only"]["all_seeds_reach_overfit_threshold"])
    full = bool(summaries["lam_full_objective"]["all_seeds_reach_overfit_threshold"])
    if matched and ce_only and not full:
        return "AUXILIARY_OBJECTIVE_INTERFERENCE"
    if matched and not ce_only:
        return "LAM_BACKBONE_OR_LATENT_PATH_TRAINABILITY_FAILURE"
    if not matched:
        return "SHARED_ENCODER_OR_OPTIMIZATION_TRAINABILITY_FAILURE"
    if matched and ce_only and full:
        return "TRAINABILITY_PASSES_COLLAPSE_IS_GENERALIZATION_OR_DATA_OBJECTIVE_INTERACTION"
    return "MIXED_OR_SEED_SENSITIVE_TRAINABILITY_FAILURE"


def close(actual: object, expected: float, label: str) -> None:
    value = float(actual)
    require(math.isfinite(value), f"{label}: non-finite")
    require(math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9), f"{label}: mismatch {value} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the train-only ARC collapse diagnostic.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    require(payload.get("artifact_type") == "LAM-JEPA ARC trainability overfit diagnostic", "wrong artifact type")
    require(payload.get("scope") == "training-only diagnostic; no validation or test access", "scope drift")
    require(payload.get("protocol_context") == "lam-jepa-arc-challenge-v4", "wrong protocol context")

    subset = payload.get("subset") or {}
    require(subset.get("source") == "ARC train only", "diagnostic source changed")
    require(subset.get("rows") == 32 and subset.get("per_class") == 8, "diagnostic subset size drift")
    require(subset.get("label_counts") == {"0": 8, "1": 8, "2": 8, "3": 8}, "diagnostic subset is not balanced")
    ids = subset.get("ids")
    require(isinstance(ids, list) and len(ids) == 32 and len(set(ids)) == 32, "diagnostic subset IDs invalid")

    training = payload.get("training") or {}
    require(training.get("seeds") == [1, 2], "diagnostic seeds drift")
    require(training.get("steps") == 300, "diagnostic step budget drift")
    require(training.get("checkpoints") == CHECKPOINTS, "checkpoint schedule drift")
    require(training.get("batch_size") == 32, "diagnostic must be full-batch on the balanced subset")
    close(training.get("learning_rate"), 0.0003, "learning rate")
    require(training.get("model_steps") == 1, "model_steps drift")
    close(training.get("overfit_threshold"), THRESHOLD, "overfit threshold")

    capacity = payload.get("capacity") or {}
    lam_active = int(capacity.get("lam_gradient_active_parameters", 0))
    matched = int(capacity.get("matched_trainable_parameters", 0))
    require(lam_active > 0 and matched > 0, "capacity evidence missing")
    ratio = matched / lam_active
    close(capacity.get("matched_ratio_to_lam_active"), ratio, "capacity ratio")
    require(0.99 <= ratio <= 1.01, f"matched diagnostic capacity outside 1%: {ratio}")

    conditions = payload.get("conditions") or {}
    expected_names = ["matched_supervised", "lam_ce_only", "lam_full_objective"]
    require(sorted(conditions) == sorted(expected_names), "diagnostic condition set changed")
    recomputed: dict[str, dict[str, object]] = {}
    for name in expected_names:
        records = conditions.get(name)
        require(isinstance(records, list) and [record.get("seed") for record in records] == [1, 2], f"{name}: seed records drift")
        for record in records:
            history = record.get("history")
            require(isinstance(history, list) and [point.get("step") for point in history] == CHECKPOINTS, f"{name}/seed={record.get('seed')}: history drift")
            for point in history:
                accuracy = float(point.get("accuracy"))
                ce = float(point.get("cross_entropy"))
                max_prob = float(point.get("mean_max_probability"))
                logit_var = float(point.get("mean_logit_variance_across_examples"))
                require(0.0 <= accuracy <= 1.0, f"{name}: invalid accuracy")
                require(math.isfinite(ce) and ce >= 0.0, f"{name}: invalid CE")
                require(0.25 <= max_prob <= 1.0, f"{name}: invalid max probability")
                require(math.isfinite(logit_var) and logit_var >= 0.0, f"{name}: invalid logit variance")
                require(1 <= int(point.get("unique_predicted_classes")) <= 4, f"{name}: invalid prediction support")
                if name.startswith("lam_"):
                    for key in ("latent_summary_feature_std_mean", "z_feature_std_mean", "z_q_feature_std_mean"):
                        value = float(point.get(key))
                        require(math.isfinite(value) and value >= 0.0, f"{name}: invalid {key}")
            final = history[-1]
            if name == "lam_full_objective":
                for key in ("training_total_loss", "training_supervised_loss", "training_auxiliary_loss", "gradient_norm_preclip", "choice_head_gradient_norm_preclip", "encoder_gradient_norm_preclip"):
                    require(math.isfinite(float(final.get(key))), f"{name}: missing final {key}")
        recomputed[name] = summarize(records)

    declared_summaries = payload.get("summaries") or {}
    for name in expected_names:
        declared = declared_summaries.get(name) or {}
        expected = recomputed[name]
        require(declared.get("final_accuracy_by_seed") == expected["final_accuracy_by_seed"], f"{name}: final accuracies drift")
        require(declared.get("best_accuracy_by_seed") == expected["best_accuracy_by_seed"], f"{name}: best accuracies drift")
        close(declared.get("mean_final_accuracy"), float(expected["mean_final_accuracy"]), f"{name}: mean final")
        close(declared.get("mean_best_accuracy"), float(expected["mean_best_accuracy"]), f"{name}: mean best")
        require(declared.get("all_seeds_reach_overfit_threshold") is expected["all_seeds_reach_overfit_threshold"], f"{name}: threshold flag drift")

    diagnosis = classify(recomputed)
    require(payload.get("diagnosis") == diagnosis, f"diagnosis mismatch: {payload.get('diagnosis')} != {diagnosis}")
    boundary = payload.get("claim_boundary") or {}
    require(boundary == {"validation_accessed": False, "test_accessed": False, "performance_claim_authorized": False, "mechanism_claim_authorized": False, "research_complete": False}, "claim boundary weakened")

    report = {
        "verdict": "ARC_TRAINABILITY_DIAGNOSTIC_VERIFIED",
        "diagnosis": diagnosis,
        "summaries": recomputed,
        "capacity_ratio": ratio,
        "balanced_train_rows": 32,
        "validation_accessed": False,
        "test_accessed": False,
        "performance_claim_authorized": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
