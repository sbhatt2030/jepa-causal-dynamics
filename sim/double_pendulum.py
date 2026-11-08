"""Template for a torque-driven, planar double-pendulum simulator.

Angles are absolute, measured in radians from the downward vertical.  The state
order is ``[theta_1, theta_2, omega_1, omega_2]``.  The implementation below
deliberately leaves numerical integration as the next implementation task while
making the data contract and continuous-time dynamics unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray: TypeAlias = NDArray[np.float64]
ScheduleValue: TypeAlias = float | list[float] | NDArray[np.float64]


def _at_step(value: ScheduleValue, step: int, name: str) -> float:
    """Return a scheduled value, extending a finite schedule by its last item."""
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return float(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a scalar or a non-empty 1-D array")
    return float(values[min(step, values.size - 1)])


@dataclass(frozen=True)
class DoublePendulumParameters:
    """Time-invariant physical parameters in SI units.

    Keep masses, rod lengths, and gravity fixed within a rollout.  Parameter
    sweeps should create one instance per trajectory; changing inertial
    geometry mid-trajectory describes a different physical system.
    """

    mass_1: float = 1.0
    mass_2: float = 1.0
    length_1: float = 1.0
    length_2: float = 1.0
    gravity: float = 9.81
    damping_1: float = 0.0
    damping_2: float = 0.0


@dataclass(frozen=True)
class InputSchedules:
    """Exogenous inputs sampled at integration steps.

    Each field accepts a scalar or an independent 1-D array.  A short array is
    held constant after its final entry, so schedules never need equal lengths.
    ``torque_1`` acts at the first pivot; ``torque_2`` acts at the second joint.
    Viscous friction is a fixed system property and belongs in
    ``DoublePendulumParameters``, not in this intervention signal.
    """

    torque_1: ScheduleValue = 0.0
    torque_2: ScheduleValue = 0.0
    def at_step(self, step: int) -> tuple[float, float]:
        if step < 0:
            raise ValueError("step must be non-negative")
        return (
            _at_step(self.torque_1, step, "torque_1"),
            _at_step(self.torque_2, step, "torque_2"),
        )


@dataclass(frozen=True)
class IntegrationConfig:
    """Numerical/output controls, not physical parameters."""

    dt: float = 0.01
    steps: int = 1_000
    method: str = "RK4"  # "RK4", "RK45", or "DOP853"
    rtol: float = 1e-7
    atol: float = 1e-9
    output_stride: int = 1  # retain every nth integration-step state


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory consumed by learning code and animation code."""

    time: FloatArray  # shape (T,)
    state: FloatArray  # shape (T, 4): theta_1, theta_2, omega_1, omega_2
    inputs: FloatArray  # shape (T, 2): tau_1, tau_2


def state_derivative(
    state: FloatArray,
    parameters: DoublePendulumParameters,
    inputs: tuple[float, float],
) -> FloatArray:
    """Evaluate the continuous-time equations of motion.

    The accelerations solve ``M(q) q_ddot = rhs``.  This function is complete
    and is the one equation implementation to validate before choosing an ODE
    solver.
    """
    theta_1, theta_2, omega_1, omega_2 = np.asarray(state, dtype=np.float64)
    tau_1, tau_2 = inputs
    m1, m2 = parameters.mass_1, parameters.mass_2
    l1, l2, g = parameters.length_1, parameters.length_2, parameters.gravity
    delta = theta_1 - theta_2

    mass_matrix = np.array(
        [[(m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(delta)],
         [m2 * l1 * l2 * np.cos(delta), m2 * l2**2]],
        dtype=np.float64,
    )
    rhs = np.array(
        [
            tau_1 - parameters.damping_1 * omega_1
            - m2 * l1 * l2 * np.sin(delta) * omega_2**2
            - (m1 + m2) * g * l1 * np.sin(theta_1),
            tau_2 - parameters.damping_2 * omega_2
            + m2 * l1 * l2 * np.sin(delta) * omega_1**2
            - m2 * g * l2 * np.sin(theta_2),
        ],
        dtype=np.float64,
    )
    acceleration = np.linalg.solve(mass_matrix, rhs)
    return np.array([omega_1, omega_2, *acceleration], dtype=np.float64)


def potential_energy(state: FloatArray, parameters: DoublePendulumParameters) -> FloatArray:
    """Return gravitational potential energy for one state or an array of states."""
    states = np.asarray(state, dtype=np.float64)
    if states.shape[-1] != 4:
        raise ValueError("state must have final dimension 4")
    theta_1, theta_2 = np.moveaxis(states[..., :2], -1, 0)
    m1, m2 = parameters.mass_1, parameters.mass_2
    l1, l2, g = parameters.length_1, parameters.length_2, parameters.gravity
    return -(m1 + m2) * g * l1 * np.cos(theta_1) - m2 * g * l2 * np.cos(theta_2)


def kinetic_energy(state: FloatArray, parameters: DoublePendulumParameters) -> FloatArray:
    """Return kinetic energy for one state or an array of states."""
    states = np.asarray(state, dtype=np.float64)
    if states.shape[-1] != 4:
        raise ValueError("state must have final dimension 4")
    theta_1, theta_2, omega_1, omega_2 = np.moveaxis(states, -1, 0)
    m1, m2 = parameters.mass_1, parameters.mass_2
    l1, l2 = parameters.length_1, parameters.length_2
    return 0.5 * (
        (m1 + m2) * l1**2 * omega_1**2
        + 2.0 * m2 * l1 * l2 * np.cos(theta_1 - theta_2) * omega_1 * omega_2
        + m2 * l2**2 * omega_2**2
    )


def total_energy(state: FloatArray, parameters: DoublePendulumParameters) -> FloatArray:
    """Return total mechanical energy for one state or an array of states.

    The potential reference is arbitrary; downward rest has the minimum
    potential energy. The returned value is in consistent physical units.
    """
    return kinetic_energy(state, parameters) + potential_energy(state, parameters)


def relative_energy(state: FloatArray, parameters: DoublePendulumParameters) -> FloatArray:
    """Energy above downward rest, normalized by the gravitational energy scale."""
    potential_minimum = -(parameters.mass_1 + parameters.mass_2) * parameters.gravity * parameters.length_1 - parameters.mass_2 * parameters.gravity * parameters.length_2
    scale = -potential_minimum
    if scale <= 0:
        raise ValueError("gravity and pendulum lengths must define a positive energy scale")
    return (total_energy(state, parameters) - potential_minimum) / scale


def simulate_double_pendulum(
    initial_state: FloatArray,
    parameters: DoublePendulumParameters,
    schedules: InputSchedules,
    config: IntegrationConfig,
) -> SimulationResult:
    """Integrate a rollout with fixed RK4 or adaptive SciPy Runge--Kutta.

    ``RK4`` takes exactly one classical fourth-order step of size ``dt`` per
    output interval.  ``RK45`` and ``DOP853`` use ``solve_ivp`` internally on
    each interval, while returning the same uniformly sampled output grid.
    Torque is held constant over every interval ``[t_k, t_(k+1)]``.
    """
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if initial_state.shape != (4,) or not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must be a finite array with shape (4,)")
    if config.dt <= 0:
        raise ValueError("dt must be positive")
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.output_stride <= 0:
        raise ValueError("output_stride must be positive")
    if config.method not in {"RK4", "RK45", "DOP853"}:
        raise ValueError("method must be 'RK4', 'RK45', or 'DOP853'")

    time = config.dt * np.arange(config.steps + 1)
    state = np.empty((config.steps + 1, 4), dtype=np.float64)
    inputs = np.empty((config.steps + 1, 2), dtype=np.float64)
    state[0] = initial_state

    for step in range(config.steps):
        torque = schedules.at_step(step)
        inputs[step] = torque
        current_state = state[step]

        if config.method == "RK4":
            k1 = state_derivative(current_state, parameters, torque)
            k2 = state_derivative(
                current_state + 0.5 * config.dt * k1, parameters, torque
            )
            k3 = state_derivative(
                current_state + 0.5 * config.dt * k2, parameters, torque
            )
            k4 = state_derivative(current_state + config.dt * k3, parameters, torque)
            state[step + 1] = current_state + (config.dt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        else:
            solution = solve_ivp(
                fun=lambda _time, y: state_derivative(y, parameters, torque),
                t_span=(time[step], time[step + 1]),
                y0=current_state,
                method=config.method,
                t_eval=[time[step + 1]],
                rtol=config.rtol,
                atol=config.atol,
            )
            if not solution.success:
                raise RuntimeError(f"integration failed at step {step}: {solution.message}")
            state[step + 1] = solution.y[:, -1]

    # The terminal state has no following interval, so retain the final applied torque.
    inputs[-1] = inputs[-2]
    output_indices = np.arange(0, config.steps + 1, config.output_stride)
    if output_indices[-1] != config.steps:
        output_indices = np.append(output_indices, config.steps)
    return SimulationResult(
        time=time[output_indices],
        state=state[output_indices],
        inputs=inputs[output_indices],
    )
