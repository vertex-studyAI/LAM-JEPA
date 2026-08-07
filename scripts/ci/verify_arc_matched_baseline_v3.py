from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def independently_verify_source(path: Path, declared: object, *, required_choice_count: int) -> list:
    require(isinstance(declared, dict), "source eligibility evidence missing")
    source = load_arc_split(path)
    eligible = [example for example in source if len(example.choices) == required_choice_count]
    excluded = [example for example in source if len(example.choices) != required_choice_count]
    distribution = Counter(len(example.choices) for example in source)
    require(int(declared.get("source_rows", -1)) == len(source), "source row count mismatch")
    require(declared.get("source_dataset_digest") == dataset_digest(source), "source dataset digest mismatch")
    require(declared.get("source_id_digest") == id_digest(source), "source ID digest mismatch")
    require({int(k): int(v) for k, v in (declared.get("choice_count_distribution") or {}).items()} == dict(sorted(distribution.items())), "choice-count distribution mismatch")
    require(int(declared.get("eligible_rows", -1)) == len(eligible), "eligible row count mismatch")
    require(declared.get("eligible_dataset_digest") == dataset_digest(eligible), "eligible dataset digest mismatch")
    require(declared.get("eligible_id_digest") == id_digest(eligible), "eligible ID digest mismatch")
    require(int(declared.get("excluded_rows", -1)) == len(excluded), "excluded row count mismatch")
    require(declared.get("excluded_id_digest") == id_digest(excluded), "excluded ID digest mismatch")
    require(
        declared.get("excluded") == [{"id": example.item_id, "choice_count": len(example.choices)} for example in excluded],
        "excluded row evidence mismatch",
    )
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind matched ARC execution to frozen protocol v3 eligibility/budget.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base-verification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-stage", choices=["development_smoke", "validation_stage"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    base = json.loads(args.base_verification.read_text(encoding="utf-8"))
    frozen = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(base.get("verdict") == "CAPACITY_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY", "base matched verifier did not pass")
    require(base.get("locked_test_evaluated") is False, "base verifier reports locked-test access")
    require(frozen.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong frozen protocol")
    eligibility = (frozen.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "protocol-v3 choice-count contract drift")

    protocol = payload.get("protocol") or {}
    require(protocol.get("protocol_id") == frozen["protocol_id"], "result protocol id mismatch")
    require(protocol.get("run_stage") == args.expected_stage, "run stage mismatch")
    require(protocol.get("eligibility_rule") == eligibility.get("rule"), "eligibility rule mismatch")
    require(protocol.get("eligibility_applied_before_limits") is True, "eligibility-before-limit assertion missing")
    require(protocol.get("test_split_accessed") is False, "wrapper reports test access")
    require(protocol.get("test_split_policy") == "not downloaded or evaluated by this development command", "test boundary weakened")

    train_eligible = independently_verify_source(args.train, protocol.get("train_source_eligibility"), required_choice_count=required_choice_count)
    validation_eligible = independently_verify_source(args.validation, protocol.get("validation_source_eligibility"), required_choice_count=required_choice_count)
    train_n = int(protocol.get("train_examples", 0))
    validation_n = int(protocol.get("validation_examples", 0))
    require(0 < train_n <= len(train_eligible), "invalid used train count")
    require(0 < validation_n <= len(validation_eligible), "invalid used validation count")
    require(protocol.get("train_digest") == dataset_digest(train_eligible[:train_n]), "matched run train digest is not eligible-prefix digest")
    require(protocol.get("train_id_digest") == id_digest(train_eligible[:train_n]), "matched run train ID digest is not eligible-prefix digest")
    require(protocol.get("validation_digest") == dataset_digest(validation_eligible[:validation_n]), "matched run validation digest is not eligible-prefix digest")
    require(protocol.get("validation_id_digest") == id_digest(validation_eligible[:validation_n]), "matched run validation ID digest is not eligible-prefix digest")
    require(not ({e.item_id for e in train_eligible} & {e.item_id for e in validation_eligible}), "eligible train/validation leakage")

    lam_active = int(protocol.get("lam_gradient_active_parameters", 0))
    matched_active = int(protocol.get("matched_supervised_gradient_active_parameters", 0))
    require(lam_active > 0 and matched_active > 0, "matched capacity evidence missing")
    ratio = matched_active / lam_active
    matched_contract = ((frozen.get("models") or {}).get("matched_capacity_supervised_baseline") or {})
    lower = float(matched_contract.get("allowed_parameter_ratio_min", 0.0))
    upper = float(matched_contract.get("allowed_parameter_ratio_max", 0.0))
    require(lower <= ratio <= upper, f"v3 matched-capacity ratio failed: {ratio:.9f}")
    require(float(protocol.get("parameter_match_tolerance", 1.0)) <= max(1.0 - lower, upper - 1.0) + 1e-12, "runner parameter tolerance weaker than v3")
    require(float(protocol.get("wall_clock_seconds", 0.0)) > 0.0, "wall-clock evidence missing")
    expected_steps = math.ceil(train_n / int(protocol["batch_size"])) * int(protocol["epochs"])
    require(int(protocol.get("optimization_steps_per_model_per_seed", -1)) == expected_steps, "optimization-step evidence mismatch")

    if args.expected_stage == "validation_stage":
        budget = frozen.get("training_budget") or {}
        require(protocol.get("seeds") == budget.get("training_seeds") == [1, 2, 3, 4, 5], "validation-stage seed budget drift")
        require(int(protocol.get("epochs", 0)) == int(budget.get("epochs", 0)) == 20, "validation-stage epoch budget drift")
        require(int(protocol.get("batch_size", 0)) == int(budget.get("batch_size", 0)) == 32, "validation-stage batch budget drift")
        require(math.isclose(float(protocol.get("learning_rate", 0.0)), float(budget.get("matched_baseline_learning_rate", -1.0)), rel_tol=0.0, abs_tol=1e-12), "matched validation LR drift")
        require(int(protocol.get("model_steps", 0)) == int(budget.get("model_steps", 0)) == 1, "validation-stage planner-step drift")
        require(train_n == len(train_eligible), "validation-stage run did not use all eligible train rows")
        require(validation_n == len(validation_eligible), "validation-stage run did not use all eligible validation rows")

    report = {
        "verdict": "PROTOCOL_V3_MATCHED_BASELINE_EXECUTION_VERIFIED_ONLY",
        "protocol_id": frozen["protocol_id"],
        "run_stage": args.expected_stage,
        "seeds": protocol.get("seeds"),
        "train_source_rows": len(load_arc_split(args.train)),
        "train_eligible_rows": len(train_eligible),
        "train_used_rows": train_n,
        "validation_source_rows": len(load_arc_split(args.validation)),
        "validation_eligible_rows": len(validation_eligible),
        "validation_used_rows": validation_n,
        "lam_gradient_active_parameters": lam_active,
        "matched_gradient_active_parameters": matched_active,
        "parameter_ratio": ratio,
        "optimization_steps_per_model_per_seed": expected_steps,
        "locked_test_evaluated": False,
        "research_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
