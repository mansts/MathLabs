import numpy as np
from src.double_pendulum import simulate, mechanical_energy
def test_energy_drift_decreases_with_smaller_timestep():
    initial_state = np.array([1.0, 0.0, 1.5, 0.0])
    errors = []
    for dt in (0.004, 0.002):
        _, states = simulate(
            initial_state,
            duration=5.0,
            dt=dt,
        )
        energy = mechanical_energy(states)
        drift = np.max(np.abs(energy - energy[0]))
        errors.append(drift)
    assert errors[1] < errors[0], (
        f"Energy drift did not decrease: {errors}"
    )
