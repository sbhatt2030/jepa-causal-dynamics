"""Generate the trajectory consumed by ``render_double_pendulum.py``.

Run from the project root:
    .venv\\Scripts\\python examples/generate_example_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.double_pendulum import (
    DoublePendulumParameters,
    InputSchedules,
    IntegrationConfig,
    simulate_double_pendulum,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "example_trajectory.npz"


def main() -> None:
    """Simulate a damped release with a brief, known first-joint torque pulse."""
    config = IntegrationConfig(dt=0.01, steps=3_000, method="DOP853")
    torque_1 = np.zeros(config.steps, dtype=np.float64)
    torque_1[100:150] = 1.0  # 0.5-second intervention, from 1.0 to 1.5 seconds.

    parameters = DoublePendulumParameters(damping_1=0.01, damping_2=0.01)
    result = simulate_double_pendulum(
        initial_state=np.array([0, -3.14, -1.0, 1.0]),
        parameters=parameters,
        schedules=InputSchedules(torque_1=torque_1),
        config=config,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        time=result.time,
        state=result.state,
        inputs=result.inputs,
        length_1=parameters.length_1,
        length_2=parameters.length_2,
    )
    print(f"Wrote {OUTPUT_PATH} with {result.time.size} states.")


if __name__ == "__main__":
    main()
