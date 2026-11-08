"""Stage-0 conservative double-pendulum dataset generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.double_pendulum import (
    DoublePendulumParameters,
    InputSchedules,
    IntegrationConfig,
    relative_energy,
    simulate_double_pendulum,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Stage0DatasetConfig:
    """Specification for a reproducible conservative dataset.

    Total energy is a truncated normal. ``energy_min`` and ``energy_max`` are
    hard stops; energy uses the scale documented by ``relative_energy``.
    """

    seed: int = 0
    episodes: int = 100
    duration: float = 20.0
    dt: float = 0.001
    output_stride: int = 10
    energy_mean: float = 1.2
    energy_std: float = 0.30
    energy_min: float = 0.7
    energy_max: float = 1.7
    potential_fraction_alpha: float = 2.0


def sample_initial_states(
    config: Stage0DatasetConfig,
    parameters: DoublePendulumParameters,
) -> FloatArray:
    """Draw exact-energy states without angle/velocity rejection sampling.

    First draw total energy and a kinetic/potential split. One angle is chosen
    randomly and the other is solved from potential energy. A Cholesky solve
    then places velocity on the exact kinetic-energy ellipse.
    """
    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    if config.energy_std <= 0 or config.energy_min >= config.energy_max:
        raise ValueError("invalid truncated-normal energy parameters")
    if config.potential_fraction_alpha <= 0:
        raise ValueError("potential_fraction_alpha must be positive")
    rng = np.random.default_rng(config.seed)
    energies: list[float] = []
    while len(energies) < config.episodes:
        draw = rng.normal(config.energy_mean, config.energy_std, size=config.episodes)
        energies.extend(draw[(draw >= config.energy_min) & (draw <= config.energy_max)])
    energies_array = np.asarray(energies[: config.episodes], dtype=np.float64)

    a = (parameters.mass_1 + parameters.mass_2) * parameters.gravity * parameters.length_1
    b = parameters.mass_2 * parameters.gravity * parameters.length_2
    energy_scale = a + b
    potential_minimum = -energy_scale
    states = np.empty((config.episodes, 4), dtype=np.float64)
    for index, energy in enumerate(energies_array):
        potential_fraction = rng.beta(config.potential_fraction_alpha, config.potential_fraction_alpha)
        target_potential = potential_minimum + potential_fraction * energy * energy_scale
        kinetic = (1.0 - potential_fraction) * energy * energy_scale
        coefficients = (a, b)
        free_index = int(rng.integers(0, 2))
        solved_index = 1 - free_index
        free_coefficient, solved_coefficient = coefficients[free_index], coefficients[solved_index]
        # V = -a cos(theta_1) - b cos(theta_2). Sample the free cosine only
        # where the solved cosine remains in [-1, 1].
        residual = -target_potential
        cosine_low = max(-1.0, (residual - solved_coefficient) / free_coefficient)
        cosine_high = min(1.0, (residual + solved_coefficient) / free_coefficient)
        if cosine_low > cosine_high:  # Defensive: should be impossible for 0 <= e <= 2.
            raise RuntimeError("target potential has no feasible angle configuration")
        free_cosine = rng.uniform(cosine_low, cosine_high)
        solved_cosine = (residual - free_coefficient * free_cosine) / solved_coefficient
        angles = np.empty(2, dtype=np.float64)
        angles[free_index] = rng.choice((-1.0, 1.0)) * np.arccos(np.clip(free_cosine, -1.0, 1.0))
        angles[solved_index] = rng.choice((-1.0, 1.0)) * np.arccos(np.clip(solved_cosine, -1.0, 1.0))

        delta = angles[0] - angles[1]
        mass_matrix = np.array(
            [[(parameters.mass_1 + parameters.mass_2) * parameters.length_1**2,
              parameters.mass_2 * parameters.length_1 * parameters.length_2 * np.cos(delta)],
             [parameters.mass_2 * parameters.length_1 * parameters.length_2 * np.cos(delta),
              parameters.mass_2 * parameters.length_2**2]],
            dtype=np.float64,
        )
        velocity_angle = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(velocity_angle), np.sin(velocity_angle)])
        cholesky = np.linalg.cholesky(mass_matrix)
        omega = np.linalg.solve(cholesky.T, np.sqrt(2.0 * kinetic) * direction)
        states[index] = [angles[0], angles[1], omega[0], omega[1]]
    return states


def _batch_stage0_derivative(state: FloatArray, parameters: DoublePendulumParameters) -> FloatArray:
    """Evaluate unforced Stage-0 dynamics for every episode in one array call."""
    theta_1, theta_2, omega_1, omega_2 = state.T
    m1, m2 = parameters.mass_1, parameters.mass_2
    l1, l2, gravity = parameters.length_1, parameters.length_2, parameters.gravity
    delta = theta_1 - theta_2
    matrix_11 = (m1 + m2) * l1**2
    matrix_12 = m2 * l1 * l2 * np.cos(delta)
    matrix_22 = m2 * l2**2
    rhs_1 = -m2 * l1 * l2 * np.sin(delta) * omega_2**2 - (m1 + m2) * gravity * l1 * np.sin(theta_1)
    rhs_2 = m2 * l1 * l2 * np.sin(delta) * omega_1**2 - m2 * gravity * l2 * np.sin(theta_2)
    determinant = matrix_11 * matrix_22 - matrix_12**2
    acceleration_1 = (matrix_22 * rhs_1 - matrix_12 * rhs_2) / determinant
    acceleration_2 = (matrix_11 * rhs_2 - matrix_12 * rhs_1) / determinant
    return np.column_stack((omega_1, omega_2, acceleration_1, acceleration_2))


def _simulate_stage0_batch(
    initial_state: FloatArray,
    parameters: DoublePendulumParameters,
    integration: IntegrationConfig,
) -> tuple[FloatArray, FloatArray]:
    """Integrate Stage-0 episodes together with the existing fixed RK4 rule."""
    output_indices = np.arange(0, integration.steps + 1, integration.output_stride)
    if output_indices[-1] != integration.steps:
        output_indices = np.append(output_indices, integration.steps)
    state = np.array(initial_state, copy=True)
    output = np.empty((state.shape[0], output_indices.size, 4), dtype=np.float64)
    output[:, 0] = state
    output_position = 1
    for step in range(integration.steps):
        k1 = _batch_stage0_derivative(state, parameters)
        k2 = _batch_stage0_derivative(state + 0.5 * integration.dt * k1, parameters)
        k3 = _batch_stage0_derivative(state + 0.5 * integration.dt * k2, parameters)
        k4 = _batch_stage0_derivative(state + integration.dt * k3, parameters)
        state += (integration.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if output_position < output_indices.size and step + 1 == output_indices[output_position]:
            output[:, output_position] = state
            output_position += 1
    return integration.dt * output_indices, output


def generate_stage0_dataset(
    config: Stage0DatasetConfig,
    parameters: DoublePendulumParameters | None = None,
) -> dict[str, FloatArray]:
    """Generate unforced, undamped episodes from a single deterministic seed."""
    parameters = parameters or DoublePendulumParameters(damping_1=0.0, damping_2=0.0)
    if parameters.damping_1 != 0.0 or parameters.damping_2 != 0.0:
        raise ValueError("stage 0 requires zero damping")
    steps = round(config.duration / config.dt)
    if steps <= 0:
        raise ValueError("duration / dt must yield at least one integration step")
    integration = IntegrationConfig(
        dt=config.dt, steps=steps, method="RK4", output_stride=config.output_stride
    )
    initial_state = sample_initial_states(config, parameters)
    time, state = _simulate_stage0_batch(initial_state, parameters, integration)
    inputs = np.zeros((*state.shape[:2], 2), dtype=np.float64)
    return {
        "time": time,
        "state": state,
        "inputs": inputs,
        "initial_state": initial_state,
        "initial_relative_energy": relative_energy(initial_state, parameters),
    }
