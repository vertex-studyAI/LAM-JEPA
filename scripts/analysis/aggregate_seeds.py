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

from lam_jepa.analysis.statistics import summarize_seed_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-seed run outputs into a reproducible summary.")
    parser.add_argument("--runs-dir", type=str, default="experiments")
    parser.add_argument("--out", type=str, default="experiments/aggregate/summary.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    by_task: dict[str, list[float]] = {}
    for path in sorted(runs_dir.glob("seed_*/results.json")):
        data = json.loads(path.read_text())
        for task, metrics in data.get("scores", {}).items():
            by_task.setdefault(task, []).append(float(metrics.get("accuracy", 0.0)))
    payload = summarize_seed_runs(by_task)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
