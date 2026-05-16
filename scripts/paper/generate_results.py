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

from lam_jepa.benchmarking.edtech_suite import seed_sweep
from lam_jepa.analysis.statistics import bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and seed summaries.")
    parser.add_argument("--out-dir", type=str, default="papers")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = seed_sweep(seeds=args.seeds, steps=80, batch_size=32, device="cpu")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "supplementary").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables" / "seed_summary.json").write_text(json.dumps(payload["aggregate"], indent=2), encoding="utf-8")
    (out_dir / "supplementary" / "seed_records.json").write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
