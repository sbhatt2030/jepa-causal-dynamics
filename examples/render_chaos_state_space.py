"""Render the chaos ensemble in unwrapped angle-state space.

Each coloured point represents one double pendulum at ``(theta_1, theta_2)``;
its trail is the path taken through that two-dimensional state-space projection.

Run after ``generate_chaos_ensemble.py``:
    .venv\\Scripts\\manim -pql examples/render_chaos_state_space.py ChaosStateSpace
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import BLUE, GREEN, ORANGE, PINK, PURPLE, RED, YELLOW, Axes, Dot, Scene, Text, TracedPath, ValueTracker, always_redraw, linear

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "artifacts" / "chaos_ensemble.npz"
COLORS = [BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]
ANIMATION_SECONDS = 60.0


def padded_range(values: np.ndarray) -> tuple[float, float]:
    """Return plot limits with a small margin, including degenerate data."""
    low, high = float(values.min()), float(values.max())
    margin = max(0.08 * (high - low), 0.5)
    return low - margin, high + margin


class ChaosStateSpace(Scene):
    def construct(self) -> None:
        data = np.load(TRAJECTORY_PATH)
        time = data["time"]
        state = data["state"]  # shape: (ensemble_size, T, 4)
        if time.ndim != 1 or state.ndim != 3 or state.shape[1:] != (time.size, 4):
            raise ValueError("expected time=(T,) and state=(ensemble_size, T, 4)")

        theta_1, theta_2 = state[:, :, 0], state[:, :, 1]
        x_low, x_high = padded_range(theta_1)
        y_low, y_high = padded_range(theta_2)
        axes = Axes(
            x_range=[x_low, x_high, np.pi],
            y_range=[y_low, y_high, np.pi],
            x_length=7.5,
            y_length=6.5,
            tips=False,
            axis_config={"include_numbers": False},
        )
        labels = axes.get_axis_labels(Text("theta_1"), Text("theta_2"))
        self.add(axes, labels)

        clock = ValueTracker(float(time[0]))

        def state_point_now(index: int) -> np.ndarray:
            return axes.c2p(
                np.interp(clock.get_value(), time, theta_1[index]),
                np.interp(clock.get_value(), time, theta_2[index]),
            )

        for index in range(state.shape[0]):
            color = COLORS[index % len(COLORS)]
            self.add(
                TracedPath(
                    lambda index=index: state_point_now(index),
                    stroke_color=color,
                    stroke_width=1.5,
                ),
                always_redraw(
                    lambda index=index, color=color: Dot(
                        state_point_now(index), color=color, radius=0.06
                    )
                ),
            )

        self.play(
            clock.animate.set_value(float(time[-1])),
            run_time=ANIMATION_SECONDS,
            rate_func=linear,
        )
