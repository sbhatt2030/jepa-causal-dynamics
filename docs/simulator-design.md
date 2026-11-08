# Double-pendulum simulator design

## Scope

The simulator produces physically interpretable controlled trajectories for the
JEPA datasets. Its output is independent of visualization:

```text
time:   (T,)
state:  (T, 4)  [theta_1, theta_2, omega_1, omega_2]
inputs: (T, 2)  [tau_1, tau_2]
```

This makes the `inputs` field the intervention/conditioning signal and lets a
renderer consume `state` without being part of the simulation loop.

## Coordinates and physics

Both angles are **absolute angles from vertical-down**, in radians. The first
rod joins the fixed pivot to mass `m1`; the second joins `m1` to mass `m2`.
Let `q = (theta_1, theta_2)`, `delta = theta_1 - theta_2`, and use rod lengths
`l1`, `l2`. With generalized torques `tau_1`, `tau_2` and viscous damping
coefficients `b1`, `b2`, the governing equation is:

```text
M(q) q_ddot = r(q, q_dot, u)

M = [[(m1+m2) l1^2,       m2 l1 l2 cos(delta)],
     [m2 l1 l2 cos(delta), m2 l2^2            ]]

r1 = tau1 - b1 omega1 - m2 l1 l2 sin(delta) omega2^2
     - (m1+m2) g l1 sin(theta1)
r2 = tau2 - b2 omega2 + m2 l1 l2 sin(delta) omega1^2
     - m2 g l2 sin(theta2)
```

Then `state_dot = [omega1, omega2, q_ddot_1, q_ddot_2]`. The code in
`sim/double_pendulum.py` implements this derivative.

## Parameter policy

`DoublePendulumParameters` holds geometry, masses, gravity, and viscous
friction coefficients as constants for one rollout. Those values change the
system itself, so use separate trajectories for parameter sweeps.  Viscous
friction produces the joint torque `-damping_i * omega_i` and is not an
intervention signal in the baseline.

`InputSchedules` holds time-varying exogenous torque inputs. Each torque
accepts a scalar or a non-empty 1-D array; a schedule shorter than a rollout
retains its last entry. Thus arrays may have different lengths. At step `k`,
all inputs are constant over `[k*dt, (k+1)*dt)`.

`IntegrationConfig` controls only numerical/output precision: time step,
solver tolerances, solver method, and downsampling. It must not be treated as a
physical intervention.

## Implementation sequence

1. `simulate_double_pendulum` supports fixed-step `RK4` and adaptive SciPy
   `RK45`/`DOP853`, using `state_derivative` as its RHS. Each method holds the
   torque constant during a schedule interval and returns the same uniform
   output grid.
2. Add validation and tests for output shapes, a constant-input trajectory,
   and convergence when `dt` is reduced.
3. Implement a Manim `Scene` which creates two rods/two bobs and uses
   `bob_positions(result.state, l1, l2)` to update them according to time.
4. Before training, export `.npz` bundles containing state, inputs, time, and
   the static parameters. Do not export rendered pixels as the canonical data.

## Manim choice

Manim is the right optional dependency for polished, presentation-quality
animations and annotated causal-intervention clips. It should not be in the
simulation core: rendering dependencies are heavier and batch experiment
inspection is often quicker with Matplotlib. The geometry adapter leaves both
options open.

## Stage-0 data generation and splits

The Stage-0 generator draws normalized total energy from a truncated normal,
then allocates it between potential and kinetic energy. It samples either
`theta_1` or `theta_2` and solves the other angle exactly from the selected
potential energy; a Cholesky solve produces velocities at the exact remaining
kinetic energy. It produces unforced, undamped trajectories from one
deterministic seed:

```powershell
.venv\Scripts\python -m data.generate_stage0 --episodes 100 --seed 7
```

For training, use independently generated split archives rather than splitting
windows from one archive. This prevents trajectories from leaking across
partitions. The split generator derives and records one seed per split, writes
the physical and data-generation settings into every archive, and writes a
`manifest.json` containing those settings, archive checksums, and initial
energy summaries. It also creates `initial_energy_histograms.png`, with one
histogram trace per split.

```powershell
.venv\Scripts\python -m data.generate_stage0_splits --seed 7
```

Train, validation, and test share the ordinary energy distribution but are
sampled independently. `hard_validation` and `hard_test` are generated from a
disjoint higher-energy band. Use ordinary validation for in-distribution
selection, hard validation for declared out-of-distribution development checks,
and reserve hard test for final reporting only.

`examples/render_double_pendulum.py` is the rendering template. After the
example generator writes `artifacts/example_trajectory.npz` with `time`,
`state`, `inputs`, `length_1`, and `length_2`, render it with:

```powershell
.venv\Scripts\python examples/generate_example_trajectory.py
.venv\Scripts\manim -pql examples/render_double_pendulum.py DoublePendulumTrajectory
```

## Chaos validation

Use the chaos ensemble to show sensitivity to initial conditions. It creates
multiple *identical double pendulums* with zero torque and zero damping; only
`theta_1` receives a tiny, known initial perturbation. The initially
overlapping coloured trajectories should visibly diverge. The generator's
default rollout and each chaos renderer are 60 seconds, producing a one-minute
animation at the default playback speed:

```powershell
.venv\Scripts\python examples/generate_chaos_ensemble.py --count 7
.venv\Scripts\manim -pql examples/render_chaos_ensemble.py ChaosEnsemble
.venv\Scripts\manim -pql examples/render_chaos_state_space.py ChaosStateSpace
.venv\Scripts\manim -pql examples/render_chaos_phase_portraits.py ChaosPhasePortraits
.venv\Scripts\python examples/generate_chaos_ensemble.py --damping-1 0.05 --damping-2 0.05 --output artifacts/chaos_damped_ensemble.npz
.venv\Scripts\manim -pql examples/render_chaos_energy.py ChaosEnergy
```
