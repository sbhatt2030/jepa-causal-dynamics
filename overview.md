# Causal structure in latent world models: JEPAs on controlled dynamical systems

Shivam Bhatt · October 2026

## Abstract

This project investigates whether the latent representations learned by joint-embedding predictive architectures (JEPAs) recover the causal structure of the dynamical systems that generate their training data. A JEPA is trained to predict the representation of future observations rather than the observations themselves. Whether that objective yields representations organized around the system's state, its controllable inputs and its unobserved influences is largely untested in settings where ground truth is available.

The question is studied on a simulated double pendulum, where partial observability, unobserved actuation, parameter variation and the degree of chaos can be varied independently, and where the true state, Jacobians and Lyapunov spectrum are known. The project also tests whether physical inductive biases (energy conservation, dissipativity) imposed on the latent predictor improve sample efficiency and dynamical fidelity, and how they behave when unobserved factors violate them.

## Research question and hypotheses

**Question.** Does the latent space learned by an action-conditioned JEPA reflect the causal structure of the underlying system, and under what conditions (interventional data, partial observability, physical inductive biases) does it do so?

- **H1, interventions.** Latent predictors trained with known actuation recover the system's input-response structure. Predictors trained on passive observation encode spurious dependencies when unobserved inputs act on several degrees of freedom.
- **H2, partial observability.** When degrees of freedom are unobserved, a JEPA recovers more of the hidden state than a reconstruction-based forecaster of matched capacity, and the gap depends on the system's largest Lyapunov exponent.
- **H3, physical priors.** A physically constrained latent predictor improves sample efficiency and long-horizon dynamical fidelity when the imposed law holds. When unobserved inputs violate the law, the constraint degrades prediction, and its residual becomes informative about those inputs.
- **H4, expectation violation.** Latent prediction error discriminates physically impossible trajectories from rare but admissible ones, and physical constraints sharpen that discrimination.

## Motivation

**Partial observability.** In systems such as neural populations or turbulent flows, an observer resolves only a fraction of the relevant degrees of freedom. Eliminating unresolved variables yields effective dynamics with memory and apparent noise (the Mori-Zwanzig formalism; Zwanzig, 2001). Unlike measurement noise, this apparent noise is temporally structured, so part of the hidden state is in principle recoverable from observation history. How much a learned model recovers is an empirical question.

**Prediction in representation space.** A reconstruction-based model is penalized for every unpredictable component of its target and converges toward conditional means. A JEPA defines its own target representation and can omit unpredictable components; its objective favors predictive information, the information the past carries about the future (Bialek et al., 2001). This freedom is the method's main advantage and its main risk: nothing guarantees that the retained information is the causally relevant information.

**Observation versus intervention.** Passive prediction captures statistical dependence, not causation. An unobserved input acting on several state variables induces dependencies between them that a passively trained predictor will encode. Action conditioning supplies interventional data for the actuated variables (Pearl, 2009). Causal representation learning asks when learned latents identify the true causal variables (Schölkopf et al., 2021); this project examines that question for JEPA objectives specifically.

**Possibility as reachability.** A model that distinguishes possible from impossible futures implicitly represents the reachable set: the states attainable from the current state under admissible inputs. Violation-of-expectation paradigms probe this capacity in infants (Baillargeon et al., 1985) and have recently been applied to self-supervised video models (Garrido et al., 2025). Latent prediction error provides the corresponding signal in a JEPA.

## Physical inductive bias

The value of built-in structure is contested. The "bitter lesson" (Sutton, 2019) holds that general methods leveraging computation outperform built-in knowledge as scale grows, and the history of handcrafted features in computer vision supports it. This project draws a distinction between approximate handcrafted features and exact properties of the data-generating process. Conservation laws, symmetries and dissipation inequalities hold wherever the system is physically consistent. Translation equivariance in convolutional networks is a precedent for exact structure persisting at scale (Bronstein et al., 2021), and Hamiltonian and Lagrangian neural networks improve long-horizon behavior when trained on raw states (Greydanus et al., 2019; Cranmer et al., 2020). Their effect on predictors operating in a learned latent space, which has no fixed physical coordinates, is less understood.

Rather than assume either position, the project measures it: prediction error as a function of dataset size, with and without the constraint, under equal hyperparameter-optimization budgets. A persistent advantage, or better generalization to held-out energies and parameters, supports structured models for this system class. A vanishing advantage supports the general approach.

Three design choices follow:

- **Soft penalty versus architectural constraint.** Only the latter strictly restricts the hypothesis class, and constraint terms are known to complicate optimization in physics-informed networks (Krishnapriyan et al., 2021). Sample efficiency and optimization speed are therefore reported separately.
- **Dissipativity under actuation.** With inputs, energy is not conserved. The relevant constraint is that a learned storage function on the latent state increases by no more than the supplied power (Willems, 1972).
- **Prior versus procedure.** Physical inductive bias constrains what the model can represent; curriculum learning and hyperparameter scheduling shape how it is trained. The two are ablated independently.

## Method

**Architecture.** A context encoder maps a window of n observations to a latent vector. A target encoder of the same architecture, updated as an exponential moving average (EMA) of the context encoder and receiving no gradients, encodes a window shifted k steps ahead (Grill et al., 2020; Assran et al., 2023). An action-conditioned predictor maps the context latent and the actuation sequence over the prediction gap to a predicted target latent. The loss is the squared error between predicted and target latents, plus variance and covariance regularization to prevent representational collapse (Bardes et al., 2022). Training includes single-step and multi-step latent rollouts. Angles are encoded as (sin θ, cos θ) to remove the discontinuity at ±π.

**Baseline.** A reconstruction-based forecaster with matched encoder capacity predicts the future observation window directly, with its loss in observation space. Comparing the two isolates the effect of predicting in representation space.

**Evaluation protocol.** The JEPA objective is defined relative to a learned target, so its loss is not comparable across runs and is trivially minimized by a collapsed representation. All results are read through probes trained on frozen encoders using held-out data, never through the training loss.

## Testbed

A planar double pendulum is simulated from its Lagrangian equations of motion with an adaptive integrator, then subsampled to the observation rate so that integration accuracy is decoupled from sampling frequency. The system is written in state-space form:

$$
x_{t+1} = f(x_t,\ u_t,\ d_t) + w_t \qquad y_t = h(x_t) + v_t
$$

The state $x$ comprises both joint angles and angular velocities; mass and length ratios are drawn per episode and held constant within it. Here $u$ is known joint torque, $d$ is unobserved torque, $w$ is process noise, $h$ is the observation map (full state or proximal link only) and $v$ is measurement noise. Each component is enabled independently. Observations and known inputs form the training data; the true state, unobserved inputs and system energy are stored separately as ground truth.

The degree of chaos is controlled by initial energy: low-energy motion is near-linear about the stable equilibrium, and high-energy motion is chaotic. In unactuated experiments, held-out test sets are defined by energy bands. Energy levels are invariant sets without damping, and under damping energy decreases monotonically, so training trajectories never enter a higher held-out band. Analytic Jacobians and the Lyapunov spectrum of the true system are computed for comparison with the learned predictor.

## Experimental program

Each stage enables one additional component of the simulator, so that every effect has a single attributable cause.

0. **Fully observed autonomous system.** Full-state observation, no actuation, no noise. Establishes stable training, collapse diagnostics and the forecaster baseline.
1. **Known actuation.** Joint torques are applied and recorded. Tests whether the predictor learns the input-response map (H1).
2. **Latent system parameters.** Mass and length ratios vary per episode. Tests whether parameters are inferred from context and encoded in the latent.
3. **Partial observability.** Only the proximal link is observed. Tests recovery of the distal link's state (H2).
4. **Unobserved actuation.** Torques act but are not recorded. Tests whether passive training induces spurious dependencies, and whether interventional data removes them (H1).
5. **Expectation violation.** Test trajectories contain non-physical events: energy discontinuities, state jumps, parameter switches within an episode (H4).

Each stage begins in the low-energy, low-noise regime; initial energy and noise levels are then swept. Physically constrained predictors (energy-conserving in the unactuated, undamped case; dissipativity-constrained under actuation) run alongside every stage to test H3, including stages in which unobserved inputs violate the imposed law.

## Evaluation

| Property | Metric |
| --- | --- |
| Representational collapse | Per-dimension latent variance; effective rank; identity-predictor baseline |
| State decodability | Linear and shallow MLP probes from latent to true state; randomly initialized encoder as control |
| Predictive accuracy | Probe-decoded multi-step latent rollouts against simulated trajectories; error vs. horizon |
| Dynamical fidelity | Eigenvalues of the predictor Jacobian at the stable equilibrium vs. those of the true k-step map; largest Lyapunov exponent at high energy |
| Causal structure | Predictor Jacobian with respect to the action input; response of latent subspaces to actuation of each link; spurious dependencies under unobserved actuation |
| Hidden-variable recovery | Probe accuracy for the unobserved link's state and per-episode parameters |
| Expectation violation | Latent prediction error at non-physical events vs. at rare admissible events (ROC) |
| Sample efficiency | Probe-decoded error vs. dataset size, with equal hyperparameter-optimization budget per configuration |

Eigenvalues and Lyapunov exponents are invariant under smooth invertible reparameterization, which permits comparison between learned and true dynamics despite arbitrary latent coordinates. This holds only where probes show that the latent retains the full state, so decodability is established first.

## Status and roadmap

Ongoing since October 2026. Problem formulation and experimental design are complete; implementation begins with the simulator. No results are reported yet.

| Milestone | Scope | Status |
| --- | --- | --- |
| Simulator and data pipeline | n-link pendulum with switchable actuation, observation map and noise; ground-truth export | Next |
| Baseline JEPA and forecaster | Stage 0; collapse diagnostics; probes | Planned |
| Actuation and causal structure | Stages 1 and 4 | Planned |
| Hidden structure | Stages 2 and 3 | Planned |
| Physical priors | Constrained predictors across stages | Planned |
| Expectation violation | Stage 5 | Planned |

Longer-term extensions: pixel observations from rendered frames, connecting to video JEPAs; model-predictive control in the learned latent space; and generalization analysis of latent dynamics parameterized by properties of the generating system (Lyapunov exponents, mixing time), informed by the sample-efficiency measurements.

## References

- Assran, M., et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR*.
- Baillargeon, R., Spelke, E. S., & Wasserman, S. (1985). Object permanence in five-month-old infants. *Cognition*, 20(3).
- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. *ICLR*.
- Bialek, W., Nemenman, I., & Tishby, N. (2001). Predictability, complexity, and learning. *Neural Computation*, 13(11).
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*.
- Cranmer, M., et al. (2020). Lagrangian neural networks. *ICLR Workshop on Integration of Deep Neural Models and Differential Equations*.
- Garrido, Q., et al. (2025). Intuitive physics understanding emerges from self-supervised pretraining on natural videos. *arXiv*.
- Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian neural networks. *NeurIPS*.
- Grill, J.-B., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS*.
- Krishnapriyan, A. S., et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Schölkopf, B., et al. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5).
- Sutton, R. (2019). The bitter lesson. Essay, incompleteideas.net.
- Willems, J. C. (1972). Dissipative dynamical systems, Part I: General theory. *Archive for Rational Mechanics and Analysis*, 45(5).
- Zwanzig, R. (2001). *Nonequilibrium Statistical Mechanics*. Oxford University Press.
