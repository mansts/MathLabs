# Research-1 — Double Pendulum Reliability Horizon

**MathLabs research project**

This repository studies how parameter uncertainty affects the predictability of a
double pendulum. The first milestone is a transparent baseline simulator. Later
milestones will add sensitivity equations, Monte Carlo experiments, Lyapunov
analysis, convergence checks, and a reliability map.

## Research question

How accurately can parameter-sensitivity equations estimate the time at which
a perturbed double-pendulum trajectory first differs from a reference trajectory
by a prescribed angular tolerance?

## Current status

- [x] Baseline double-pendulum equations (ideal, frictionless model)
- [x] Fixed-step classical RK4 integrator
- [x] Mechanical-energy calculation for numerical validation
- [x] Baseline trajectory experiment and energy-drift plot
- [ ] Parameter-sensitivity equations
- [ ] Monte Carlo reference distribution and convergence study
- [ ] Lyapunov-exponent comparison
- [ ] Reliability/applicability map

## Model assumptions (Milestone 1)

- Two point masses, connected by massless rigid rods.
- Angles are measured from the downward vertical.
- Gravity is constant.
- No friction or external forces in this first version.
- The model is deterministic; uncertainty will be introduced in later experiments.

The frictionless model is intentional: it gives us a conserved-energy check
before adding any damping model.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline experiment:

```bash
python experiments/01_baseline.py
```

The script saves `artifacts/baseline_trajectory.png` and
`artifacts/energy_drift.png`.

Run tests:

```bash
python -m pytest
```

## Reliability-horizon definition (planned)

For reference angles `theta_ref(t)` and perturbed angles `theta_p(t)`, define
the wrapped angular difference for each arm in `[-pi, pi)` and

\[
D(t)=\sqrt{(\Delta\theta_1(t))^2+(\Delta\theta_2(t))^2}.
\]

For tolerance `epsilon`, the reliability horizon is

\[
T_\epsilon=\inf\{t>0:D(t)\geq\epsilon\}.
\]

The first study will use `epsilon = 0.1 rad`. We will later specify how the
first-crossing time is interpolated between integration steps.

## Scientific integrity

Monte Carlo is a numerical reference, not mathematical ground truth. Its
sampling convergence and the integration-step error must be measured. All
reported accuracy claims will be based on recorded experiments, not assumed
in advance.
