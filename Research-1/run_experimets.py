"""Run the first sensitivity vs Monte Carlo experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.comparison import compare_methods


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    initial_state = np.array([
        1.0,  # theta1
        0.0,  # omega1
        1.5,  # theta2
        0.0,  # omega2
    ])

    result = compare_methods(
        initial_state,
        uncertainty=0.01,
        n_runs=500,
        duration=10.0,
        dt=0.002,
        epsilon=0.1,
        seed=42,
    )

    times = result["times"]
    predicted = result["predicted_distance"]
    monte_carlo = result["mc_rms_distance"]

    # Save numerical results
    csv_path = output_dir / "experiment_1.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time",
            "sensitivity_prediction",
            "monte_carlo_rms",
            "absolute_error",
            "relative_error",
        ])

        for i in range(len(times)):
            writer.writerow([
                times[i],
                predicted[i],
                monte_carlo[i],
                result["absolute_error"][i],
                result["relative_error"][i],
            ])

    # Plot comparison
    plt.figure(figsize=(10, 6))

    plt.plot(
        times,
        predicted,
        label="First-order sensitivity",
    )

    plt.plot(
        times,
        monte_carlo,
        label="Monte Carlo RMS",
    )

    plt.axhline(
        0.1,
        linestyle="--",
        label="Reliability threshold",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular distance (radians)")
    plt.title("Sensitivity vs Monte Carlo")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "comparison.png",
        dpi=200,
    )

    plt.close()

    # Save summary
    summary = {
        "experiment": "Sensitivity vs Monte Carlo",
        "uncertainty": result["uncertainty"],
        "n_runs": result["n_runs"],
        "predicted_horizon": result["predicted_horizon"],
        "mc_median_horizon": result["mc_median_horizon"],
        "mc_fraction_reached": result["mc_fraction_reached"],
        "initial_state": initial_state.tolist(),
        "duration": 10.0,
        "dt": 0.002,
        "epsilon": 0.1,
    }

    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=4)

    print("\nEXPERIMENT COMPLETED")
    print("--------------------")
    print(f"Monte Carlo runs: {result['n_runs']}")
    print(f"Predicted horizon: {result['predicted_horizon']}")
    print(f"MC median horizon: {result['mc_median_horizon']}")
    print(f"MC threshold fraction: {result['mc_fraction_reached']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
