"""Ideal double-pendulum model and fixed-step RK4 integrator.

Angles are measured from the downward vertical. State order:
(theta1, omega1, theta2, omega2).
"""

from __future__ import annotations

import numpy as np


def derivatives(
    state: np.ndarray,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Return time derivatives for the frictionless double pendulum."""
    theta1, omega1, theta2, omega2 = np.asarray(state, dtype=float)
    delta = theta1 - theta2
    denominator = 2.0 * m1 + m2 - m2 * np.cos(2.0 * delta)

    alpha1 = (
        -g * (2.0 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2.0 * theta2)
        - 2.0
        * m2
        * np.sin(delta)
        * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    ) / (l1 * denominator)

    alpha2 = (
        2.0
        * np.sin(delta)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(delta)
        )
    ) / (l2 * denominator)

    return np.array([omega1, alpha1, omega2, alpha2], dtype=float)


def rk4_step(
    state: np.ndarray,
    dt: float,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Advance the state by one classical fourth-order Runge-Kutta step."""
    f = lambda y: derivatives(y, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(
    initial_state: np.ndarray,
    *,
    duration: float = 10.0,
    dt: float = 0.001,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate and return (times, states), including the initial state."""
       
    if not np.isfinite(duration) or not np.isfinite(dt):
        raise ValueError("duration and dt must be finite.")

    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive.")

    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (4,):
        raise ValueError(
            "initial_state must contain exactly four values: "
            "(theta1, omega1, theta2, omega2)."
        )

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain only finite values.")
    if min(m1, m2, l1, l2, g) <= 0:
        raise ValueError("masses, lengths, and gravity must be positive")

    steps = int(np.ceil(duration / dt))
    times = np.arange(steps + 1, dtype=float) * dt
    times[-1] = min(times[-1], duration)
    states = np.empty((steps + 1, 4), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)

    for i in range(steps):
        step = min(dt, duration - times[i])
        states[i + 1] = rk4_step(
            states[i], step, m1=m1, m2=m2, l1=l1, l2=l2, g=g
        )

    return times, states


def mechanical_energy(
    states: np.ndarray,
    *,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> np.ndarray:
    """Return total mechanical energy (kinetic + potential)."""
    values = np.asarray(states, dtype=float)
    theta1, omega1, theta2, omega2 = values.T

    kinetic = (
        0.5 * (m1 + m2) * (l1 * omega1) ** 2
        + 0.5 * m2 * (l2 * omega2) ** 2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    potential = (
        -(m1 + m2) * g * l1 * np.cos(theta1)
        - m2 * g * l2 * np.cos(theta2)
    )
    return kinetic + potential
