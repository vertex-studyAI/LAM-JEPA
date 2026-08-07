from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

CHECKPOINTS = [0, 1, 5, 10, 25, 50, 100, 200, 300]
CONDITIONS = ["canonical", "quantizer_only", "no_quantizer", "direct_latent"]
THRESHOLD = 0.95
EXPECTED_CONFIGS = {
    "canonical": {"use_quantizer": True, "use_memory": True, "use_planner": True},
    "quantizer_only": {"use_quantizer": True, "use_memory": False, "use_planner": False},
    "no_quantizer": {"use_quantizer": False, "use_memory": True, "use_planner": True},
    "direct_latent": {"use_quantizer": False, "use_memory": False, "use_planner": False},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def finite(value, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label}: non-finite")
    return number


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
    canonical = bool(summaries["canonical"]["all_seeds_reach_overfit_threshold"])
    quantizer_only = bool(summaries["quantizer_only"]["all_seeds_reach_overfit_threshold"])
    no_quantizer = bool(summaries["no_quantizer"]["all_seeds_reach_overfit_threshold"])
    direct = bool(summaries["direct_latent"]["all_seeds_reach_overfit_threshold"])
    if canonical:
        return "CANONICAL_TRAINABILITY_NOW_PASSES"
    if direct and no_quantizer and not quantizer_only:
        return "QUANTIZER_CAUSAL_COLLAPSE_SUPPORTED"
    if direct and not no_quantizer:
        return "POST_QUANTIZER_MEMORY_OR_PLANNER_FAILURE_REMAINS"
    if not direct:
        return "DIRECT_LATENT_OR_ANSWER_HEAD_TRAINABILITY_FAILURE"
    return "MIXED_OR_SEED_SENSITIVE_QUANTIZER_CAUSAL_RESULT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify train-only ARC quantizer causal ablation evidence.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    require(payload.get("artifact_type") == "LAM-JEPA ARC quantizer causal ablation", "wrong artifact type")
    require(payload.get("scope") == "training-only diagnostic; no validation or test access", "scope drift")
    require(payload.get("protocol_context") == "lam-jepa-arc-challenge-v4", "wrong protocol context")

    subset = payload.get("subset") or {}
    require(subset.get("source") == "ARC train only", "wrong source")
    require(subset.get("rows") == 32 and subset.get("per_class") == 8, "subset drift")
    require(subset.get("label_counts") == {"0": 8, "1": 8, "2": 8, "3": 8}, "subset not balanced")
    ids = subset.get("ids")
    require(isinstance(ids, list) and len(ids) == 32 and len(set(ids)) == 32, "invalid subset ids")

    training = payload.get("training") or {}
    require(training.get("seeds") == [1, 2], "seed drift")
    require(training.get("steps") == 300, "step budget drift")
    require(training.get("checkpoints") == CHECKPOINTS, "checkpoint drift")
    require(training.get("batch_size") == 32, "batch size drift")
    require(training.get("model_steps") == 1, "model_steps drift")
    require(math.isclose(float(training.get("learning_rate")), 0.0003, rel_tol=0, abs_tol=1e-12), "learning rate drift")
    require(math.isclose(float(training.get("overfit_threshold")), THRESHOLD, rel_tol=0, abs_tol=1e-12), "threshold drift")

    conditions = payload.get("conditions") or {}
    require(sorted(conditions) == sorted(CONDITIONS), "condition set drift")
    recomputed: dict[str, dict[str, object]] = {}
    for name in CONDITIONS:
        records = conditions.get(name)
        require(isinstance(records, list) and [record.get("seed") for record in records] == [1, 2], f"{name}: seed records drift")
        for record in records:
            require(record.get("config") == EXPECTED_CONFIGS[name], f"{name}: config drift")
            history = record.get("history")
            require(isinstance(history, list) and [point.get("step") for point in history] == CHECKPOINTS, f"{name}: history drift")
            for point in history:
                accuracy = finite(point.get("accuracy"), f"{name}: accuracy")
                require(0.0 <= accuracy <= 1.0, f"{name}: invalid accuracy")
                require(finite(point.get("cross_entropy"), f"{name}: CE") >= 0.0, f"{name}: invalid CE")
                require(1 <= int(point.get("unique_predicted_classes")) <= 4, f"{name}: invalid prediction support")
                require(0.25 <= finite(point.get("mean_max_probability"), f"{name}: confidence") <= 1.0, f"{name}: invalid confidence")
                for key in (
                    "mean_logit_variance_across_examples",
                    "z_feature_std_mean",
                    "z_q_feature_std_mean",
                    "latent_summary_feature_std_mean",
                    "quant_loss",
                ):
                    require(finite(point.get(key), f"{name}: {key}") >= 0.0, f"{name}: invalid {key}")
                if int(point.get("step")) > 0:
                    for key in (
                        "training_loss",
                        "gradient_norm_preclip",
                        "encoder_gradient_norm_preclip",
                        "projector_gradient_norm_preclip",
                        "quantizer_gradient_norm_preclip",
                        "choice_head_gradient_norm_preclip",
                    ):
                        require(finite(point.get(key), f"{name}: {key}") >= 0.0, f"{name}: invalid {key}")
        recomputed[name] = summarize(records)

    declared = payload.get("summaries") or {}
    for name in CONDITIONS:
        require(declared.get(name) == recomputed[name], f"{name}: summary mismatch")

    diagnosis = classify(recomputed)
    require(payload.get("diagnosis") == diagnosis, "diagnosis mismatch")
    require(
        payload.get("claim_boundary")
        == {
            "validation_accessed": False,
            "test_accessed": False,
            "performance_claim_authorized": False,
            "mechanism_claim_authorized_beyond_trainability": False,
            "research_complete": False,
        },
        "claim boundary weakened",
    )

    report = {
        "verdict": "ARC_QUANTIZER_CAUSAL_ABLATION_VERIFIED",
        "diagnosis": diagnosis,
        "summaries": recomputed,
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
