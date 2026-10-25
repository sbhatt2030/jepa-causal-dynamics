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
| Simulator and data pipeline | Next |
| Baseline JEPA and reconstruction forecaster (Stage 0) | Planned |
| Actuation and causal structure (Stages 1, 4) | Planned |
| Hidden structure (Stages 2, 3) | Planned |
| Physical priors (all stages) | Planned |
| Expectation violation (Stage 5) | Planned |

## Repository structure

Planned layout; directories are added as each milestone is implemented.

```
sim/       Lagrangian n-link pendulum simulator, ground-truth export, Lyapunov spectrum
data/      Dataset generation; splits by energy band and system parameters
models/    JEPA (context encoder, EMA target encoder, predictor) and reconstruction baseline
eval/      Probes, rollout metrics, Jacobian eigenvalue and Lyapunov analysis
configs/   Experiment configurations
tests/     Unit tests
docs/      Project overview
```

## Author

Shivam Bhatt, University of Waterloo
