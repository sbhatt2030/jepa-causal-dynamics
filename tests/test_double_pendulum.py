import numpy as np
import pytest

from sim.double_pendulum import (
    DoublePendulumParameters,
    IntegrationConfig,
    InputSchedules,
    simulate_double_pendulum,
    state_derivative,
    total_energy,
)


def test_short_schedules_hold_their_final_value() -> None:
    schedule = InputSchedules(torque_1=[1.0, 2.0], torque_2=[3.0])
    assert schedule.at_step(0) == (1.0, 3.0)
    assert schedule.at_step(8) == (2.0, 3.0)


def test_empty_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        InputSchedules(torque_1=[]).at_step(0)


def test_downward_equilibrium_has_zero_derivative() -> None:
    derivative = state_derivative(
        np.zeros(4), DoublePendulumParameters(), (0.0, 0.0)
    )
    np.testing.assert_allclose(derivative, 0.0)


def test_viscous_friction_removes_kinetic_energy() -> None:
    """At zero angle, the damping contribution must have negative power."""
    parameters = DoublePendulumParameters(gravity=0.0, damping_1=3.0, damping_2=5.0)
    state = np.array([0.0, 0.0, 2.0, -3.0])
    acceleration = state_derivative(state, parameters, (0.0, 0.0))[2:]
    delta = state[0] - state[1]
    mass_matrix = np.array(
        [
            [(parameters.mass_1 + parameters.mass_2) * parameters.length_1**2,
             parameters.mass_2 * parameters.length_1 * parameters.length_2 * np.cos(delta)],
            [parameters.mass_2 * parameters.length_1 * parameters.length_2 * np.cos(delta),
             parameters.mass_2 * parameters.length_2**2],
        ]
    )
    # d/dt (1/2 omega.T M omega) = omega.T M q_ddot when q = 0.
    assert state[2:] @ mass_matrix @ acceleration < 0.0


def test_total_energy_is_constant_for_an_unforced_undamped_rollout() -> None:
    result = simulate_double_pendulum(
        initial_state=np.array([1.3, -1.3, 0.1, -0.1]),
        parameters=DoublePendulumParameters(),
        schedules=InputSchedules(),
        config=IntegrationConfig(dt=0.001, steps=1_000, method="RK4"),
    )
    energy = total_energy(result.state, DoublePendulumParameters())
    np.testing.assert_allclose(energy, energy[0], rtol=2e-8, atol=1e-8)


@pytest.mark.parametrize("method", ["RK4", "RK45", "DOP853"])
def test_all_integrators_preserve_downward_rest(method: str) -> None:
    result = simulate_double_pendulum(
        initial_state=np.zeros(4),
        parameters=DoublePendulumParameters(),
        schedules=InputSchedules(),
        config=IntegrationConfig(dt=0.01, steps=5, method=method),
    )
    assert result.state.shape == (6, 4)
    assert result.inputs.shape == (6, 2)
    np.testing.assert_allclose(result.state, 0.0)
    np.testing.assert_allclose(result.inputs, 0.0)


def test_terminal_input_repeats_last_applied_torque() -> None:
    result = simulate_double_pendulum(
        initial_state=np.zeros(4),
        parameters=DoublePendulumParameters(),
        schedules=InputSchedules(torque_1=[1.0, 2.0]),
        config=IntegrationConfig(steps=2),
    )
    np.testing.assert_allclose(result.inputs[:, 0], [1.0, 2.0, 2.0])

def test_rk4_agrees_with_high_accuracy_reference() -> None:
    initial_state = np.array([0.4, -0.2, 0.0, 0.1])
    parameters = DoublePendulumParameters(damping_1=0.05, damping_2=0.05)
    schedules = InputSchedules()

    rk4 = simulate_double_pendulum(
        initial_state,
        parameters,
        schedules,
        IntegrationConfig(dt=0.001, steps=1_000, method="RK4"),
    )
    reference = simulate_double_pendulum(
        initial_state,
        parameters,
        schedules,
        IntegrationConfig(
            dt=0.001,
            steps=1_000,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        ),
    )

    np.testing.assert_allclose(
        rk4.state,
        reference.state,
        rtol=1e-4,
        atol=1e-5,
    )
