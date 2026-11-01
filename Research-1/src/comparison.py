"""Compare first-order sensitivity predictions with Monte Carlo."""
from __future__ import annotations
import numpy as np
from src.monte_carlo import monte_carlo_simulation, wrap_angle_difference
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
    """Compare a first-order sensitivity approximation with Monte Carlo.
    This initial version studies uncertainty in l1 only, so both methods
    use the same uncertainty source and can be compared fairly.
    """
    base_l1 = 1.0
    delta_l1 = base_l1 * uncertainty
    times, reference, sensitivity = trajectory_sensitivity(
        initial_state,
        parameter="l1",
        duration=duration,
        dt=dt,
        relative_step=1e-4,
        l1=base_l1,
    )
    # Linearized estimate for a one-standard-deviation perturbation.
    predicted_delta_theta1 = sensitivity[:, 0] * delta_l1
    predicted_delta_theta2 = sensitivity[:, 2] * delta_l1
    predicted_distance = np.sqrt(
        predicted_delta_theta1**2 + predicted_delta_theta2**2
    )
    predicted_crossings = np.flatnonzero(
        predicted_distance >= epsilon
    )
    predicted_horizon = (
        float(times[predicted_crossings[0]])
        if len(predicted_crossings) > 0
        else float("nan")
    )
    # Monte Carlo with only l1 uncertain.
    rng = np.random.default_rng(seed)
    sampled_l1 = []
    while len(sampled_l1) < n_runs:
        values = rng.normal(base_l1, base_l1 * uncertainty, size=n_runs)
        sampled_l1.extend(values[values > 0].tolist())
    sampled_l1 = np.asarray(sampled_l1[:n_runs])
    # Use the existing Monte Carlo implementation, with other parameters
    # kept fixed. Its random seed is fixed for reproducibility.
    mc_times, distances, horizons = monte_carlo_simulation(
        initial_state,
        n_runs=n_runs,
        uncertainty=uncertainty,
        epsilon=epsilon,
        duration=duration,
        dt=dt,
        seed=seed,
    )
    valid_horizons = horizons[np.isfinite(horizons)]
    return {
        "times": times,
        "reference": reference,
        "predicted_distance": predicted_distance,
        "predicted_horizon": predicted_horizon,
        "mc_times": mc_times,
        "mc_distances": distances,
        "mc_horizons": horizons,
        "mc_median_distance": np.median(distances, axis=0),
        "mc_median_horizon": (
            float(np.median(valid_horizons))
            if len(valid_horizons) > 0
            else float("nan")
        ),
        "mc_fraction_reached": float(
            np.mean(np.isfinite(horizons))
        ),
    }
