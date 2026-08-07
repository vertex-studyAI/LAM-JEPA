from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "benchmark"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_arc_matched_baseline as base
from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest
from lam_jepa.benchmarking.arc_protocol import ARC_PROTOCOL_CHOICE_COUNT, select_protocol_eligible_examples


PROTOCOL_ID = "lam-jepa-arc-challenge-v3"


def pop_custom_arg(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def cli_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    return sys.argv[index + 1]


def partition_evidence(examples) -> dict[str, object]:
    partition = select_protocol_eligible_examples(examples)
    return {
        "source_rows": partition.original_count,
        "source_dataset_digest": dataset_digest(examples),
        "source_id_digest": id_digest(examples),
        "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
        "choice_count_distribution": {str(key): value for key, value in partition.choice_count_distribution.items()},
        "eligible_rows": partition.eligible_count,
        "eligible_dataset_digest": dataset_digest(partition.eligible),
        "eligible_id_digest": partition.eligible_id_digest,
        "excluded_rows": partition.excluded_count,
        "excluded_id_digest": partition.excluded_id_digest,
        "excluded": [
            {"id": example.item_id, "choice_count": len(example.choices)}
            for example in partition.excluded
        ],
    }


def main() -> None:
    run_stage = pop_custom_arg("--run-stage", "development_smoke")
    if run_stage not in {"development_smoke", "validation_stage"}:
        raise SystemExit("--run-stage must be development_smoke or validation_stage")
    output_arg = cli_value("--out")
    if output_arg is None:
        raise SystemExit("--out is required")
    output_path = Path(output_arg)

    original_load = base.load_arc_split
    loaded_evidence: list[dict[str, object]] = []

    def eligible_load(path: Path):
        examples = original_load(path)
        loaded_evidence.append(partition_evidence(examples))
        return list(select_protocol_eligible_examples(examples).eligible)

    base.load_arc_split = eligible_load
    started = time.perf_counter()
    try:
        base.main()
    finally:
        base.load_arc_split = original_load
    elapsed = time.perf_counter() - started

    if len(loaded_evidence) != 2:
        raise RuntimeError(f"expected exactly two source-split loads, observed {len(loaded_evidence)}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("matched baseline output protocol block missing")

    protocol.update(
        {
            "protocol_id": PROTOCOL_ID,
            "run_stage": run_stage,
            "eligibility_rule": "retain a row if and only if len(choices) == 4",
            "required_choice_count": ARC_PROTOCOL_CHOICE_COUNT,
            "train_source_eligibility": loaded_evidence[0],
            "validation_source_eligibility": loaded_evidence[1],
            "eligibility_applied_before_limits": True,
            "test_split_accessed": False,
            "wall_clock_seconds": elapsed,
            "optimization_steps_per_model_per_seed": math.ceil(int(protocol["train_examples"]) / int(protocol["batch_size"])) * int(protocol["epochs"]),
        }
    )
    if run_stage == "validation_stage":
        protocol["claim_boundary"] = (
            "Protocol-v3 validation-stage matched comparison only. Locked test remains untouched. "
            "Validation results may determine whether the frozen superiority gate is met, but cannot authorize "
            "a confirmatory test claim, external validation, or RESEARCH_COMPLETE without the remaining frozen "
            "baseline/control evidence and Lane 08 reproduction."
        )
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "run_stage": run_stage,
                "train_source_rows": loaded_evidence[0]["source_rows"],
                "train_eligible_rows": loaded_evidence[0]["eligible_rows"],
                "validation_source_rows": loaded_evidence[1]["source_rows"],
                "validation_eligible_rows": loaded_evidence[1]["eligible_rows"],
                "used_train_rows": protocol["train_examples"],
                "used_validation_rows": protocol["validation_examples"],
                "wall_clock_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
