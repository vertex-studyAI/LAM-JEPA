from __future__ import annotations

from math import sqrt
from typing import Iterable

from .simulator import State, wrap_angle


def predict_persistence(state: State, dt: float) -> State:
    del dt
    return state


def predict_constant_velocity(state: State, dt: float) -> State:
    theta1, omega1, theta2, omega2 = state
    return (
        wrap_angle(theta1 + omega1 * dt),
        omega1,
        wrap_angle(theta2 + omega2 * dt),
        omega2,
    )


def squared_error(predicted: State, target: State) -> float:
    angular_1 = wrap_angle(predicted[0] - target[0])
    velocity_1 = predicted[1] - target[1]
    angular_2 = wrap_angle(predicted[2] - target[2])
    velocity_2 = predicted[3] - target[3]
    return (
        angular_1 * angular_1
        + velocity_1 * velocity_1
        + angular_2 * angular_2
        + velocity_2 * velocity_2
    ) / 4.0


def trajectory_rmse(states: Iterable[State], *, dt: float, method: str) -> float:
    rows = list(states)
    if len(rows) < 2:
        raise ValueError("at least two states are required")
    if method == "persistence":
        predictor = predict_persistence
    elif method == "constant_velocity":
        predictor = predict_constant_velocity
    else:
        raise ValueError(f"unknown baseline method: {method}")

    errors = [
        squared_error(predictor(rows[index], dt), rows[index + 1])
        for index in range(len(rows) - 1)
    ]
    return sqrt(sum(errors) / len(errors))


def aggregate_rmse(trajectories: Iterable[list[State]], *, dt: float, method: str) -> float:
    squared_errors: list[float] = []
    if method == "persistence":
        predictor = predict_persistence
    elif method == "constant_velocity":
        predictor = predict_constant_velocity
    else:
        raise ValueError(f"unknown baseline method: {method}")

    for states in trajectories:
        for index in range(len(states) - 1):
            squared_errors.append(squared_error(predictor(states[index], dt), states[index + 1]))
    if not squared_errors:
        raise ValueError("no transitions supplied")
    return sqrt(sum(squared_errors) / len(squared_errors))
