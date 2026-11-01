"""Compare first-order sensitivity predictions with Monte Carlo."""

from __future__ import annotations

import numpy as np

from src.monte_carlo import monte_carlo_simulation
from src.sensitivity import trajectory_sensitivity


def compare_methods(
    initial_state: np.ndarray,
    *,
    uncertainty: float = 0.01,
    n_runs: int = 500,
    duration: float = 10.0,
    dt: float = 0.002,
    epsilon: float = 0.1,
    seed: int = 42,
) -> dict:
    """Compare first-order sensitivity with Monte Carlo.

    Only l1 is uncertain in both methods.
    """

    if not 0.0 < uncertainty < 1.0:
        raise ValueError("uncertainty must be between 0 and 1.")

    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    base_l1 = 1.0
    sigma_l1 = base_l1 * uncertainty

    # First-order sensitivity
    times, reference, sensitivity = trajectory_sensitivity(
        initial_state,
        parameter="l1",
        duration=duration,
        dt=dt,
        relative_step=1e-4,
        l1=base_l1,
    )

    # Predicted one-standard-deviation angular distance
    predicted_distance = sigma_l1 * np.sqrt(
        sensitivity[:, 0] ** 2
        + sensitivity[:, 2] ** 2
    )

    predicted_crossings = np.flatnonzero(
        predicted_distance >= epsilon
    )

    predicted_horizon = (
        float(times[predicted_crossings[0]])
        if len(predicted_crossings)
        else float("nan")
    )

    # Monte Carlo: ONLY l1 is uncertain
    mc_times, distances, horizons = monte_carlo_simulation(
        initial_state,
        n_runs=n_runs,
        uncertainty=uncertainty,
        epsilon=epsilon,
        duration=duration,
        dt=dt,
        seed=seed,
        l1=base_l1,
        uncertain_parameters=("l1",),
    )

    if not np.allclose(times, mc_times):
        raise RuntimeError("Time grids do not match.")

    # RMS angular distance
    mc_rms_distance = np.sqrt(
        np.mean(distances**2, axis=0)
    )

    # Error between first-order prediction and Monte Carlo
    absolute_error = np.abs(
        predicted_distance - mc_rms_distance
    )

    valid = predicted_distance > 1e-12

    relative_error = np.full_like(
        predicted_distance,
        np.nan,
    )

    relative_error[valid] = (
        absolute_error[valid] / predicted_distance[valid]
    )

    valid_horizons = horizons[np.isfinite(horizons)]

    return {
        "times": times,
        "reference": reference,
        "sensitivity": sensitivity,

        "predicted_distance": predicted_distance,
        "predicted_horizon": predicted_horizon,

        "mc_times": mc_times,
        "mc_distances": distances,
        "mc_rms_distance": mc_rms_distance,
        "mc_horizons": horizons,

        "absolute_error": absolute_error,
        "relative_error": relative_error,

        "mc_median_horizon": (
            float(np.median(valid_horizons))
            if len(valid_horizons)
            else float("nan")
        ),

        "mc_fraction_reached": float(
            np.mean(np.isfinite(horizons))
        ),

        "uncertainty": uncertainty,
        "n_runs": n_runs,
    }
