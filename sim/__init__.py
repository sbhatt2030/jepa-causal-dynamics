"""Physics simulator interfaces for controlled planar pendulums."""

from .double_pendulum import (
    DoublePendulumParameters,
    IntegrationConfig,
    InputSchedules,
    kinetic_energy,
    potential_energy,
    SimulationResult,
    relative_energy,
    simulate_double_pendulum,
    total_energy,
)

__all__ = [
    "DoublePendulumParameters",
    "IntegrationConfig",
    "InputSchedules",
    "kinetic_energy",
    "potential_energy",
    "SimulationResult",
    "relative_energy",
    "simulate_double_pendulum",
    "total_energy",
]
