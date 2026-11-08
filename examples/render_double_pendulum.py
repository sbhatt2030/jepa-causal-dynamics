"""Render an exported simulator trajectory with Manim.

Run after implementing/exporting the simulator:
    .venv\\Scripts\\manim -pql examples/render_double_pendulum.py DoublePendulumTrajectory
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import BLUE, RED, Dot, Line, Scene, ValueTracker, always_redraw, linear

from sim.animation import bob_positions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "artifacts" / "example_trajectory.npz"


class DoublePendulumTrajectory(Scene):
    def construct(self) -> None:
        """Play state data; it does not run the physical solver."""
        # Expected keys: time (T,), state (T, 4), length_1, length_2.
        data = np.load(TRAJECTORY_PATH)
        time = data["time"]
        state = data["state"]
        length_1 = float(data["length_1"])
        length_2 = float(data["length_2"])
        if time.ndim != 1 or state.shape != (time.size, 4):
            raise ValueError("expected time=(T,) and state=(T, 4)")

        clock = ValueTracker(float(time[0]))

        def positions_now() -> np.ndarray:
            current = np.array(
                [np.interp(clock.get_value(), time, state[:, column]) for column in range(4)]
            )
            positions_2d = bob_positions(current, length_1, length_2)[0]
            # Manim stores points as (x, y, z); the simulator intentionally
            # remains 2-D and renderer-independent.
            return np.column_stack((positions_2d, np.zeros(positions_2d.shape[0])))

        pivot = Dot(positions_now()[0], color=BLUE)
        rod_1 = always_redraw(lambda: Line(positions_now()[0], positions_now()[1]))
        bob_1 = always_redraw(lambda: Dot(positions_now()[1], color=RED))
        rod_2 = always_redraw(lambda: Line(positions_now()[1], positions_now()[2]))
        bob_2 = always_redraw(lambda: Dot(positions_now()[2], color=RED))
        self.add(pivot, rod_1, bob_1, rod_2, bob_2)
        self.play(
            clock.animate.set_value(float(time[-1])),
            run_time=min(float(time[-1] - time[0]), 50.0),
            rate_func=linear,
        )
