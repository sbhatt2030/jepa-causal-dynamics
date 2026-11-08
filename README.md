# Causal structure in latent world models

**JEPAs on controlled dynamical systems**

This project investigates whether joint-embedding predictive architectures (JEPAs) learn latent representations that reflect the causal structure of the dynamical systems generating their data. It uses a simulated double pendulum, where partial observability, unobserved actuation, parameter variation and the degree of chaos can be controlled independently, and where the true state, Jacobians and Lyapunov spectrum are known. It also tests whether physical inductive biases on the latent predictor (energy conservation, dissipativity) improve sample efficiency and dynamical fidelity.

The full motivation, method, experimental program and references are in **[docs/overview.md](docs/overview.md)**.

## Research question

Does the latent space learned by an action-conditioned JEPA reflect the causal structure of the underlying system, and under what conditions (interventional data, partial observability, physical inductive biases) does it do so?

- **H1, interventions.** Training with known actuation recovers input-response structure; passive training encodes spurious dependencies under unobserved inputs.
- **H2, partial observability.** A JEPA recovers more of an unobserved state than a reconstruction-based forecaster, with a gap that depends on the largest Lyapunov exponent.
- **H3, physical priors.** Physical constraints improve sample efficiency and long-horizon fidelity when the law holds, and their residuals signal unobserved inputs when it does not.
- **H4, expectation violation.** Latent prediction error discriminates physically impossible trajectories from rare but admissible ones.

## Status

Ongoing since October 2026. Experimental design is complete; implementation begins with the simulator. No results are reported yet.

| Milestone | Status |
| --- | --- |
| Simulator and data pipeline | In progress |
| Baseline JEPA and reconstruction forecaster (Stage 0) | Planned |
| Actuation and causal structure (Stages 1, 4) | Planned |
| Hidden structure (Stages 2, 3) | Planned |
| Physical priors (all stages) | Planned |
| Expectation violation (Stage 5) | Planned |

## Stage-0 dataset splits

Generate train, validation, test, and disjoint higher-energy challenge splits
with recorded seeds, generation parameters, checksums, and an initial-energy
histogram for every trajectory population:

```powershell
.venv\Scripts\python -m data.generate_stage0_splits --seed 7
```

The command writes `train.npz`, `validation.npz`, `test.npz`,
`hard_validation.npz`, `hard_test.npz`, `manifest.json`, and
`initial_energy_histograms.png` under `artifacts/stage0/`. Ordinary validation
and test are independently generated from the training energy distribution;
the hard splits use the explicitly disjoint `1.75–1.95` relative-energy band.
Use ordinary validation for in-distribution selection, hard validation for
declared out-of-distribution development checks, and reserve hard test for
final reporting only.

## Energy animation

`examples/render_n_pendulum_energy.py` is a self-contained, one-minute Manim
dashboard. It recalculates four unforced double-pendulum trajectories at render
time, animates their motion, and plots their aggregate kinetic, gravitational
potential (relative to downward rest), and total mechanical energy.

```powershell
.venv\Scripts\pip install -e .[animation]
.venv\Scripts\manim -pql examples/render_n_pendulum_energy.py NPendulumEnergy
```

For a final 60 fps export, use `-pqh -r 1920,1080 --fps 60` in place of
`-pql`. Change `PENDULUM_COUNT`, `PHYSICAL_SECONDS`, or `ANIMATION_SECONDS`
near the top of the scene to tailor the clip.

## Experiment tracking

Stage-0 dataset generation is tracked with [MLflow](https://mlflow.org/). The
generator records its configuration, summary energy metrics, and the generated
archive. By default, runs are stored locally in `mlruns/`.
The environment also includes [Optuna](https://optuna.org/) for forthcoming
hyperparameter sweeps.

```powershell
.venv\Scripts\pip install -e .
.venv\Scripts\python -m data.generate_stage0 --episodes 100 --seed 7
.venv\Scripts\mlflow ui --backend-store-uri .\mlruns
```

Open `http://127.0.0.1:5000` to inspect the run. Supply `--tracking-uri` to
use a remote tracking server, or `--no-mlflow` for an untracked generation.

## Repository structure

Planned layout; directories are added as each milestone is implemented.

```
sim/       Controlled double-pendulum simulator and ground-truth export
data/      Dataset generation, provenance manifests, and energy-band splits
models/    JEPA (context encoder, EMA target encoder, predictor) and reconstruction baseline
eval/      Probes, rollout metrics, Jacobian eigenvalue and Lyapunov analysis
configs/   Experiment configurations
tests/     Unit tests
docs/      Project overview
```

## Author

Shivam Bhatt, University of Waterloo
