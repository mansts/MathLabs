"""Milestone 1: simulate one ideal double pendulum and check energy drift."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.double_pendulum import mechanical_energy, simulate  # noqa: E402


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    # Initial condition is a starting test case, not a claim about chaos.
    initial = np.array([np.deg2rad(120), 0.0, np.deg2rad(-10), 0.0])
    times, states = simulate(initial, duration=15.0, dt=0.001)
    energy = mechanical_energy(states)
    relative_drift = (energy - energy[0]) / max(abs(energy[0]), 1e-12)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, np.rad2deg(states[:, 0]), label=r"$\theta_1$")
    ax.plot(times, np.rad2deg(states[:, 2]), label=r"$\theta_2$")
    ax.set(xlabel="Time (s)", ylabel="Angle (degrees)",
           title="Ideal double pendulum: baseline trajectory")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(artifacts / "baseline_trajectory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, relative_drift)
    ax.set(xlabel="Time (s)", ylabel="Relative energy drift",
           title="Numerical energy conservation check (RK4)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(artifacts / "energy_drift.png", dpi=160)
    plt.close(fig)

    print(f"Initial energy: {energy[0]:.10f} J (for chosen units)")
    print(f"Maximum absolute relative energy drift: "
          f"{np.max(np.abs(relative_drift)):.3e}")
    print(f"Plots saved in: {artifacts}")


if __name__ == "__main__":
    main()
