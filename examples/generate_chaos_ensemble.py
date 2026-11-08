"""Generate nearly identical double-pendulum rollouts for a chaos demonstration.

Example:
    .venv\\Scripts\\python examples/generate_chaos_ensemble.py --count 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.double_pendulum import (
    DoublePendulumParameters,
    InputSchedules,
    IntegrationConfig,
    simulate_double_pendulum,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=7, help="number of trajectories")
    parser.add_argument("--perturbation", type=float, default=1e-3, help="maximum theta_1 offset in radians")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to simulate")
    parser.add_argument("--dt", type=float, default=0.005, help="RK4 integration timestep")
    parser.add_argument("--damping-1", type=float, default=0.0, help="viscous damping at joint 1")
    parser.add_argument("--damping-2", type=float, default=0.0, help="viscous damping at joint 2")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "chaos_ensemble.npz"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 2:
        raise ValueError("count must be at least 2")
    if args.perturbation <= 0 or args.duration <= 0 or args.dt <= 0:
        raise ValueError("perturbation, duration, and dt must be positive")

    steps = round(args.duration / args.dt)
    # An energetic, asymmetric release in a visibly chaotic regime.
    base_state = np.array([2.0, -1.6, 0.0, 0.0], dtype=np.float64)
    offsets = np.linspace(-args.perturbation, args.perturbation, args.count)
    initial_states = np.repeat(base_state[None, :], args.count, axis=0)
    initial_states[:, 0] += offsets

    if args.damping_1 < 0 or args.damping_2 < 0:
        raise ValueError("damping coefficients must be non-negative")
    parameters = DoublePendulumParameters(damping_1=args.damping_1, damping_2=args.damping_2)
    config = IntegrationConfig(dt=args.dt, steps=steps, method="RK4", output_stride=2)
    trajectories = [
        simulate_double_pendulum(initial_state, parameters, InputSchedules(), config)
        for initial_state in initial_states
    ]
    state = np.stack([trajectory.state for trajectory in trajectories])
    time = trajectories[0].time
    separation = np.linalg.norm(state[-1, -1, :2] - state[0, -1, :2])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=time,
        state=state,
        initial_state=initial_states,
        perturbation=args.perturbation,
        mass_1=parameters.mass_1,
        mass_2=parameters.mass_2,
        length_1=parameters.length_1,
        length_2=parameters.length_2,
        gravity=parameters.gravity,
        damping_1=parameters.damping_1,
        damping_2=parameters.damping_2,
    )
    print(f"Wrote {args.output} with state shape {state.shape}.")
    print(f"Final first-vs-last angular separation: {separation:.6f} rad")


if __name__ == "__main__":
    main()
