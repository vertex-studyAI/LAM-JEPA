from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product

from .simulator import State

THETA1_VALUES = (-1.2, -0.8, -0.4, 0.4, 0.8, 1.2)
THETA2_VALUES = (-1.0, -0.5, 0.5, 1.0)
OMEGA1_VALUES = (-0.5, 0.0, 0.5)
OMEGA2_VALUES = (-0.5, 0.0, 0.5)


@dataclass(frozen=True)
class TrajectorySpec:
    trajectory_id: str
    initial_state: State
    split: str
    bucket: int


def canonical_initial_state(state: State) -> str:
    return ",".join(f"{value:+.6f}" for value in state)


def split_bucket(state: State) -> int:
    digest = sha256(canonical_initial_state(state).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 10


def split_name(bucket: int) -> str:
    if not 0 <= bucket <= 9:
        raise ValueError("bucket must be in [0, 9]")
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def trajectory_id(state: State) -> str:
    digest = sha256(canonical_initial_state(state).encode("utf-8")).hexdigest()
    return f"dp-{digest[:16]}"


def frozen_grid() -> list[TrajectorySpec]:
    specs: list[TrajectorySpec] = []
    for theta1, theta2, omega1, omega2 in product(
        THETA1_VALUES,
        THETA2_VALUES,
        OMEGA1_VALUES,
        OMEGA2_VALUES,
    ):
        state: State = (theta1, omega1, theta2, omega2)
        bucket = split_bucket(state)
        specs.append(
            TrajectorySpec(
                trajectory_id=trajectory_id(state),
                initial_state=state,
                split=split_name(bucket),
                bucket=bucket,
            )
        )
    specs.sort(key=lambda row: row.trajectory_id)
    return specs
