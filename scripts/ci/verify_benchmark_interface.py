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
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    result_path = Path(args.result)
    payload = json.loads(result_path.read_text())

    assert payload.get("seeds") == args.expected_seeds, (
        payload.get("seeds"),
        args.expected_seeds,
    )
    runs = payload.get("runs")
    assert isinstance(runs, list), "benchmark output must contain a runs list"
    assert len(runs) == len(args.expected_seeds), "one run is required per requested seed"
    assert [run.get("seed") for run in runs] == args.expected_seeds

    expected_tasks = set(EDTECH_TASKS)
    for run in runs:
        tasks = run.get("tasks")
        assert isinstance(tasks, dict), "each run must contain task results"
        assert set(tasks) == expected_tasks, (
            sorted(tasks),
            sorted(expected_tasks),
        )
        assert isinstance(run.get("history_tail"), list), "each run must retain training history"

    claim_boundary = payload.get("claim_boundary")
    assert isinstance(claim_boundary, str) and claim_boundary.strip(), (
        "benchmark output must retain an explicit scientific claim boundary"
    )

    report = {
        "status": "passed",
        "result": str(result_path),
        "seeds": args.expected_seeds,
        "run_count": len(runs),
        "task_count": len(expected_tasks),
        "task_names": list(EDTECH_TASKS),
        "claim_boundary_present": True,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
