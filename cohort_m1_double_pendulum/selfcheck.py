from __future__ import annotations

from math import isfinite

from .baselines import aggregate_rmse
from .dataset import frozen_grid
from .simulator import PendulumParams, render_svg, simulate


def main() -> None:
    specs = frozen_grid()
    assert len(specs) == 216
    assert len({spec.trajectory_id for spec in specs}) == 216

    counts = {name: sum(spec.split == name for spec in specs) for name in ("train", "validation", "test")}
    assert all(counts[name] > 0 for name in counts)
    assert sum(counts.values()) == 216

    params = PendulumParams()
    first = specs[0]
    a = simulate(first.initial_state, steps=32, dt=0.02, params=params)
    b = simulate(first.initial_state, steps=32, dt=0.02, params=params)
    assert a == b
    assert len(a) == 33
    assert render_svg(a, params=params) == render_svg(b, params=params)

    small_by_split: dict[str, list[list[tuple[float, float, float, float]]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for spec in specs:
        if len(small_by_split[spec.split]) < 2:
            small_by_split[spec.split].append(
                simulate(spec.initial_state, steps=16, dt=0.02, params=params)
            )

    for split, trajectories in small_by_split.items():
        assert trajectories, split
        for method in ("persistence", "constant_velocity"):
            value = aggregate_rmse(trajectories, dt=0.02, method=method)
            assert isfinite(value)
            assert value >= 0.0

    print("M1 self-check passed")
    print(f"split counts: {counts}")


if __name__ == "__main__":
    main()
