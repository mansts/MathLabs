"""Monte Carlo analysis for double-pendulum parameter uncertainty."""
from __future__ import annotations
import numpy as np
from src.double_pendulum import simulate
PARAMETERS = ("m1", "m2", "l1", "l2")
def wrap_angle_difference(delta: np.ndarray) -> np.ndarray:
    """Wrap angular differences into [-pi, pi)."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi
def monte_carlo_simulation(
    initial_state: np.ndarray,
    *,
    n_runs: int = 500,
    uncertainty: float = 0.01,
    epsilon: float = 0.1,
    duration: float = 10.0,
    dt: float = 0.002,
    seed: int = 42,
    uncertain_parameters: tuple[str, ...] = PARAMETERS,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare a reference trajectory with perturbed simulations."""
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if not 0.0 < uncertainty < 1.0:
        raise ValueError("uncertainty must be between 0 and 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not uncertain_parameters:
        raise ValueError(
            "At least one uncertain parameter is required."
        )
    if len(set(uncertain_parameters)) != len(uncertain_parameters):
        raise ValueError(
            "uncertain_parameters must not contain duplicates."
        )
    if any(name not in PARAMETERS for name in uncertain_parameters):
        raise ValueError(
            f"uncertain_parameters must be selected from {PARAMETERS}."
        )
    base_parameters = {
        "m1": m1,
        "m2": m2,
        "l1": l1,
        "l2": l2,
        "g": g,
    }
    if any(
        not np.isfinite(value) or value <= 0
        for value in base_parameters.values()
    ):
        raise ValueError("All physical parameters must be positive and finite.")
    rng = np.random.default_rng(seed)
    times, reference = simulate(
        initial_state,
        duration=duration,
        dt=dt,
        **base_parameters,
    )
    distances = np.empty((n_runs, len(times)))
    horizons = np.full(n_runs, np.nan)
    for run in range(n_runs):
        sampled_parameters = dict(base_parameters)
        for name in uncertain_parameters:
            base_value = base_parameters[name]
            while True:
                factor = rng.normal(1.0, uncertainty)
                if factor > 0:
                    break
            sampled_parameters[name] = base_value * factor
        _, trajectory = simulate(
            initial_state,
            duration=duration,
            dt=dt,
            **sampled_parameters,
        )
        delta_theta_1 = wrap_angle_difference(
            trajectory[:, 0] - reference[:, 0]
        )
        delta_theta_2 = wrap_angle_difference(
            trajectory[:, 2] - reference[:, 2]
        )
        distance = np.sqrt(
            delta_theta_1**2 + delta_theta_2**2
        )
        distances[run] = distance
        crossings = np.flatnonzero(distance >= epsilon)
        if len(crossings) > 0:
            horizons[run] = times[crossings[0]]
    return times, distances, horizons
