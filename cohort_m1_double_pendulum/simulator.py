from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

State = tuple[float, float, float, float]


@dataclass(frozen=True)
class PendulumParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    def validate(self) -> None:
        if self.m1 <= 0 or self.m2 <= 0:
            raise ValueError("masses must be positive")
        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("lengths must be positive")
        if self.g <= 0:
            raise ValueError("gravity must be positive")


def wrap_angle(x: float) -> float:
    return (x + pi) % (2.0 * pi) - pi


def derivatives(state: State, params: PendulumParams) -> State:
    """Return d/dt [theta1, omega1, theta2, omega2]."""
    params.validate()
    theta1, omega1, theta2, omega2 = state
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    delta = theta1 - theta2
    common = 2.0 * m1 + m2 - m2 * cos(2.0 * theta1 - 2.0 * theta2)
    if abs(common) < 1e-14:
        raise FloatingPointError("singular double-pendulum denominator")

    alpha1_num = (
        -g * (2.0 * m1 + m2) * sin(theta1)
        - m2 * g * sin(theta1 - 2.0 * theta2)
        - 2.0
        * sin(delta)
        * m2
        * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos(delta))
    )
    alpha1 = alpha1_num / (l1 * common)

    alpha2_num = 2.0 * sin(delta) * (
        omega1 * omega1 * l1 * (m1 + m2)
        + g * (m1 + m2) * cos(theta1)
        + omega2 * omega2 * l2 * m2 * cos(delta)
    )
    alpha2 = alpha2_num / (l2 * common)

    return (omega1, alpha1, omega2, alpha2)


def _add_scaled(a: State, b: State, scale: float) -> State:
    return tuple(x + scale * y for x, y in zip(a, b))  # type: ignore[return-value]


def rk4_step(state: State, dt: float, params: PendulumParams) -> State:
    if dt <= 0:
        raise ValueError("dt must be positive")
    k1 = derivatives(state, params)
    k2 = derivatives(_add_scaled(state, k1, 0.5 * dt), params)
    k3 = derivatives(_add_scaled(state, k2, 0.5 * dt), params)
    k4 = derivatives(_add_scaled(state, k3, dt), params)
    out = tuple(
        s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )
    theta1, omega1, theta2, omega2 = out
    return (wrap_angle(theta1), omega1, wrap_angle(theta2), omega2)


def simulate(
    initial_state: State,
    *,
    steps: int,
    dt: float,
    params: PendulumParams = PendulumParams(),
) -> list[State]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    states = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = rk4_step(state, dt, params)
        states.append(state)
    return states


def bob_positions(state: State, params: PendulumParams = PendulumParams()) -> tuple[float, float, float, float]:
    theta1, _, theta2, _ = state
    x1 = params.l1 * sin(theta1)
    y1 = -params.l1 * cos(theta1)
    x2 = x1 + params.l2 * sin(theta2)
    y2 = y1 - params.l2 * cos(theta2)
    return x1, y1, x2, y2


def render_svg(
    states: Iterable[State],
    *,
    params: PendulumParams = PendulumParams(),
    width: int = 640,
    height: int = 640,
) -> str:
    rows = list(states)
    if not rows:
        raise ValueError("at least one state is required")
    if width < 64 or height < 64:
        raise ValueError("canvas is too small")

    radius = params.l1 + params.l2
    scale = 0.43 * min(width, height) / radius
    cx, cy = width / 2.0, height / 2.0

    def px(x: float, y: float) -> tuple[float, float]:
        return cx + scale * x, cy - scale * y

    trail: list[str] = []
    for state in rows:
        _, _, x2, y2 = bob_positions(state, params)
        tx, ty = px(x2, y2)
        trail.append(f"{tx:.3f},{ty:.3f}")

    x1, y1, x2, y2 = bob_positions(rows[-1], params)
    p0 = px(0.0, 0.0)
    p1 = px(x1, y1)
    p2 = px(x2, y2)

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<polyline points="{" ".join(trail)}" fill="none" stroke="#777" stroke-width="1" opacity="0.65"/>',
            f'<line x1="{p0[0]:.3f}" y1="{p0[1]:.3f}" x2="{p1[0]:.3f}" y2="{p1[1]:.3f}" stroke="black" stroke-width="3"/>',
            f'<line x1="{p1[0]:.3f}" y1="{p1[1]:.3f}" x2="{p2[0]:.3f}" y2="{p2[1]:.3f}" stroke="black" stroke-width="3"/>',
            f'<circle cx="{p0[0]:.3f}" cy="{p0[1]:.3f}" r="4" fill="black"/>',
            f'<circle cx="{p1[0]:.3f}" cy="{p1[1]:.3f}" r="8" fill="black"/>',
            f'<circle cx="{p2[0]:.3f}" cy="{p2[1]:.3f}" r="8" fill="black"/>',
            '</svg>',
        ]
    )
