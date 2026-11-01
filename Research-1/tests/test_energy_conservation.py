import numpy as np

from src.double_pendulum import simulate, mechanical_energy


def test_energy_conservation():

    initial_state = np.array([
        1.0,
        0.0,
        1.5,
        0.0,
    ])

    errors = []

    for dt in [0.01, 0.002, 0.001]:

        times, states = simulate(
            initial_state,
            duration=10.0,
            dt=dt,
        )

        energy = mechanical_energy(states)

        relative_error = (
            np.max(np.abs(energy - energy[0]))
            / abs(energy[0])
        )

        errors.append(relative_error)

        print(
            f"dt={dt}: "
            f"relative energy error={relative_error:.10e}"
        )

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
