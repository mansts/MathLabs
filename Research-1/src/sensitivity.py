“”“Finite-difference sensitivity analysis for the double pendulum.

This is an initial numerical approximation of parameter sensitivity.
It is not yet the full variational-equation method.
“””

from future import annotations

import numpy as np

from src.double_pendulum import simulate

def trajectory_sensitivity(
initial_state: np.ndarray,
parameter: str,
*,
duration: float = 10.0,
dt: float = 0.001,
relative_step: float = 1e-4,
m1: float = 1.0,
m2: float = 1.0,
l1: float = 1.0,
l2: float = 1.0,
g: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
“”“Estimate d(state)/d(parameter) using central differences.

Returns:
    times: simulation times
    reference: unperturbed trajectory
    sensitivity: approximate derivative of the state with respect
        to the selected model parameter.
"""
parameters = {
    "m1": m1,
    "m2": m2,
    "l1": l1,
    "l2": l2,
    "g": g,
}
if parameter not in parameters:
    raise ValueError(
        f"Unknown parameter {parameter!r}. "
        f"Choose one of {tuple(parameters)}."
    )
value = parameters[parameter]
if value <= 0:
    raise ValueError("The selected parameter must be positive.")
delta = value * relative_step
if delta == 0:
    raise ValueError("The perturbation must be non-zero.")
common = {
    "duration": duration,
    "dt": dt,
    **parameters,
}
times, reference = simulate(initial_state, **common)
plus_parameters = {**common, parameter: value + delta}
minus_parameters = {**common, parameter: value - delta}
_, plus = simulate(initial_state, **plus_parameters)
_, minus = simulate(initial_state, **minus_parameters)
sensitivity = (plus - minus) / (2.0 * delta)
return times, reference, sensitivity
