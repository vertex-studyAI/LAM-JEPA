from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import CONTRACT_VERSION
from .baselines import aggregate_rmse
from .dataset import frozen_grid
from .simulator import PendulumParams, State, render_svg, simulate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path) -> None:
    fields = ["trajectory_id", "split", "bucket", "theta1", "omega1", "theta2", "omega2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in frozen_grid():
            theta1, omega1, theta2, omega2 = spec.initial_state
            writer.writerow(
                {
                    "trajectory_id": spec.trajectory_id,
                    "split": spec.split,
                    "bucket": spec.bucket,
                    "theta1": f"{theta1:.6f}",
                    "omega1": f"{omega1:.6f}",
                    "theta2": f"{theta2:.6f}",
                    "omega2": f"{omega2:.6f}",
                }
            )


def _write_trajectory(path: Path, states: list[State], dt: float) -> None:
    fields = ["step", "time_s", "theta1", "omega1", "theta2", "omega2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, state in enumerate(states):
            theta1, omega1, theta2, omega2 = state
            writer.writerow(
                {
                    "step": index,
                    "time_s": f"{index * dt:.6f}",
                    "theta1": f"{theta1:.12f}",
                    "omega1": f"{omega1:.12f}",
                    "theta2": f"{theta2:.12f}",
                    "omega2": f"{omega2:.12f}",
                }
            )


def reproduce(out: Path, *, steps: int, dt: float) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be positive")

    out.mkdir(parents=True, exist_ok=True)
    params = PendulumParams()
    specs = frozen_grid()

    split_manifest = out / "split_manifest.csv"
    _write_manifest(split_manifest)

    trajectories: dict[str, list[list[State]]] = {"train": [], "validation": [], "test": []}
    test_examples: list[tuple[str, list[State]]] = []

    for spec in specs:
        states = simulate(spec.initial_state, steps=steps, dt=dt, params=params)
        trajectories[spec.split].append(states)
        if spec.split == "test":
            test_examples.append((spec.trajectory_id, states))

    metrics: dict[str, dict[str, float]] = {}
    for method in ("persistence", "constant_velocity"):
        metrics[method] = {
            split: aggregate_rmse(rows, dt=dt, method=method)
            for split, rows in trajectories.items()
        }

    metrics_path = out / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not test_examples:
        raise RuntimeError("frozen split unexpectedly contains no test trajectories")
    test_examples.sort(key=lambda item: item[0])
    sample_id, sample_states = test_examples[0]

    sample_csv = out / "sample_test_trajectory.csv"
    _write_trajectory(sample_csv, sample_states, dt)
    sample_svg = out / "sample_test_trajectory.svg"
    sample_svg.write_text(render_svg(sample_states, params=params) + "\n", encoding="utf-8")

    counts = Counter(spec.split for spec in specs)
    artifact_paths = [split_manifest, metrics_path, sample_csv, sample_svg]
    run_manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "steps": steps,
        "dt": dt,
        "simulator": {
            "integrator": "fixed-step-rk4",
            "m1": params.m1,
            "m2": params.m2,
            "l1": params.l1,
            "l2": params.l2,
            "g": params.g,
        },
        "trajectory_count": len(specs),
        "split_counts": dict(sorted(counts.items())),
        "sample_test_trajectory_id": sample_id,
        "artifact_sha256": {path.name: _sha256(path) for path in artifact_paths},
    }

    run_manifest_path = out / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the frozen JEPA Cohort M1 artifact")
    parser.add_argument("--out", type=Path, default=Path("artifacts/cohort_m1"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    manifest = reproduce(args.out, steps=args.steps, dt=args.dt)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
