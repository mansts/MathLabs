"""Experiment 03: Monte Carlo parameter uncertainty."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monte_carlo import monte_carlo_simulation


def main():
    initial_state = np.array([
        np.deg2rad(120),
        0.0,
        np.deg2rad(-10),
        0.0,
    ])

    times, distances, horizons = monte_carlo_simulation(
        initial_state,
        n_runs=500,
        uncertainty=0.01,
        epsilon=0.1,
        duration=10.0,
        dt=0.002,
        seed=42,
    )

    median_distance = np.median(distances, axis=0)
    lower_distance = np.percentile(distances, 10, axis=0)
    upper_distance = np.percentile(distances, 90, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(
        times,
        lower_distance,
        upper_distance,
        alpha=0.25,
        label="10th–90th percentile",
    )

    ax.plot(
        times,
        median_distance,
        label="Median angular distance",
    )

    ax.axhline(
        0.1,
        linestyle="--",
        label="Reliability threshold = 0.1 rad",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular distance (rad)")
    ax.set_title("Monte Carlo: 1% parameter uncertainty")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)

    fig.savefig(
        output / "monte_carlo_1pct.png",
        dpi=160,
    )

    valid_horizons = horizons[np.isfinite(horizons)]

    print(f"Total simulations: {len(horizons)}")
    print(f"Threshold reached: {len(valid_horizons)}")
    print(
        "Threshold not reached:",
        int(np.sum(~np.isfinite(horizons))),
    )

    if len(valid_horizons) > 0:
        print(
            "Median reliability horizon (s):",
            round(float(np.median(valid_horizons)), 4),
        )

        print(
            "10th percentile horizon (s):",
            round(float(np.percentile(valid_horizons, 10)), 4),
        )

        print(
            "90th percentile horizon (s):",
            round(float(np.percentile(valid_horizons, 90)), 4),
        )
    else:
        print(
            "No trajectory reached the threshold "
            "during the simulated interval."
        )

    print("Saved artifacts/monte_carlo_1pct.png")


if __name__ == "__main__":
    main()
