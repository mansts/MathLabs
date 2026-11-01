import numpy as np

from src.double_pendulum import derivatives, mechanical_energy, simulate


def test_derivative_shape_and_finite_values():
    result = derivatives(np.array([0.2, 0.0, -0.1, 0.0]))
    assert result.shape == (4,)
    assert np.all(np.isfinite(result))


def test_simulation_includes_initial_state():
    initial = np.array([0.2, 0.0, -0.1, 0.0])
    times, states = simulate(initial, duration=0.01, dt=0.001)
    assert states.shape == (len(times), 4)
    np.testing.assert_allclose(states[0], initial)


def test_energy_drift_decreases_with_smaller_step():
    initial = np.array([0.7, 0.0, -0.4, 0.0])
    _, coarse = simulate(initial, duration=2.0, dt=0.004)
    _, fine = simulate(initial, duration=2.0, dt=0.002)
    e_coarse = mechanical_energy(coarse)
    e_fine = mechanical_energy(fine)
    drift_coarse = np.max(np.abs((e_coarse - e_coarse[0]) / e_coarse[0]))
    drift_fine = np.max(np.abs((e_fine - e_fine[0]) / e_fine[0]))
    assert drift_fine < drift_coarse
