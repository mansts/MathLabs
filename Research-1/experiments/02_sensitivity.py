“”“Experiment 02: visualize sensitivity to the first rod length.”””

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(file).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sensitivity import trajectory_sensitivity

def main():
initial_state = np.array([
np.deg2rad(120),
0.0,
np.deg2rad(-10),
0.0,
])

times, _, sensitivity = trajectory_sensitivity(
    initial_state,
    parameter="l1",
    duration=10.0,
    dt=0.001,
    relative_step=1e-4,
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    times,
    sensitivity[:, 0],
    label=r"$\partial\theta_1/\partial l_1$",
)
ax.plot(
    times,
    sensitivity[:, 2],
    label=r"$\partial\theta_2/\partial l_1$",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angular sensitivity (rad/m)")
ax.set_title("Double-pendulum sensitivity to first rod length")
ax.grid(True)
ax.legend()
fig.tight_layout()
output = ROOT / "artifacts"
output.mkdir(exist_ok=True)
fig.savefig(output / "sensitivity_l1.png", dpi=160)
print("Saved artifacts/sensitivity_l1.png")
print("Maximum |d(theta1)/d(l1)|:",
      np.max(np.abs(sensitivity[:, 0])))
print("Maximum |d(theta2)/d(l1)|:",
      np.max(np.abs(sensitivity[:, 2])))

if name == “main”:
main()
