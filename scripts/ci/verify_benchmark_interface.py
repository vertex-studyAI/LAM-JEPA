from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "src").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lam_jepa.benchmarking.edtech_suite import EDTECH_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify benchmark CLI output structure.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-evaluation-seed", type=int, required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    payload = json.loads(result_path.read_text())
    protocol = payload.get("protocol")
    assert isinstance(protocol, dict), "benchmark output must contain a protocol object"
    assert protocol.get("training_seeds") == args.expected_seeds, (
        protocol.get("training_seeds"),
        args.expected_seeds,
    )
    assert protocol.get("evaluation_seed") == args.expected_evaluation_seed
    assert protocol.get("evaluation_pairing") == (
        "identical ordered evaluation rows across training seeds"
    )

    records = payload.get("records")
    assert isinstance(records, list), "benchmark output must contain a records list"
    assert len(records) == len(args.expected_seeds), "one record is required per requested seed"
    assert [record.get("training_seed") for record in records] == args.expected_seeds
    assert all(
        record.get("evaluation_seed") == args.expected_evaluation_seed
        for record in records
    )

    expected_tasks = set(EDTECH_TASKS)
    reference_digests = payload.get("sample_digests")
    assert isinstance(reference_digests, dict)
    assert set(reference_digests) == expected_tasks

    for record in records:
        tasks = record.get("tasks")
        assert isinstance(tasks, dict), "each record must contain task results"
        assert set(tasks) == expected_tasks, (
            sorted(tasks),
            sorted(expected_tasks),
        )
        assert isinstance(record.get("history_tail"), list), (
            "each record must retain training history"
        )
        digests = {
            task: str(metrics["sample_digest"])
            for task, metrics in tasks.items()
        }
        assert digests == reference_digests, (
            "every training seed must be evaluated on the same ordered rows"
        )

    aggregate = payload.get("aggregate")
    assert isinstance(aggregate, dict), "benchmark output must contain seed-level aggregate statistics"
    assert set(aggregate) == expected_tasks

    claim_boundary = payload.get("claim_boundary")
    assert isinstance(claim_boundary, str) and claim_boundary.strip(), (
        "benchmark output must retain an explicit scientific claim boundary"
    )

    report = {
        "status": "passed",
        "result": str(result_path),
        "training_seeds": args.expected_seeds,
        "evaluation_seed": args.expected_evaluation_seed,
        "record_count": len(records),
        "task_count": len(expected_tasks),
        "task_names": list(EDTECH_TASKS),
        "identical_evaluation_rows": True,
        "aggregate_present": True,
        "claim_boundary_present": True,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
