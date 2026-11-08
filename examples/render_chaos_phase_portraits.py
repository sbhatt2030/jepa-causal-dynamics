"""Render all six pairwise state-space projections of the chaos ensemble.

Run after ``generate_chaos_ensemble.py``:
    .venv\\Scripts\\manim -pql examples/render_chaos_phase_portraits.py ChaosPhasePortraits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import BLUE, GREEN, ORANGE, PINK, PURPLE, RED, YELLOW, Axes, Dot, Scene, Text, TracedPath, ValueTracker, VGroup, always_redraw, linear

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "artifacts" / "chaos_ensemble.npz"
COLORS = [BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]
ANIMATION_SECONDS = 60.0
PROJECTIONS = (
    ("theta_1", "theta_2"),
    ("theta_1", "omega_1"),
    ("theta_1", "omega_2"),
    ("theta_2", "omega_1"),
    ("theta_2", "omega_2"),
    ("omega_1", "omega_2"),
)


def padded_range(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    margin = max(0.08 * (high - low), 0.5)
    return low - margin, high + margin


class ChaosPhasePortraits(Scene):
    def construct(self) -> None:
        data = np.load(TRAJECTORY_PATH)
        time = data["time"]
        state = data["state"]  # (ensemble_size, T, 4)
        if time.ndim != 1 or state.ndim != 3 or state.shape[1:] != (time.size, 4):
            raise ValueError("expected time=(T,) and state=(ensemble_size, T, 4)")

        variables = {
            "theta_1": state[:, :, 0],
            "theta_2": state[:, :, 1],
            "omega_1": state[:, :, 2],
            "omega_2": state[:, :, 3],
        }
        axes_by_projection: list[Axes] = []
        panels = VGroup()
        for x_label, y_label in PROJECTIONS:
            x_low, x_high = padded_range(variables[x_label])
            y_low, y_high = padded_range(variables[y_label])
            axes = Axes(
                x_range=[x_low, x_high, (x_high - x_low) / 4.0],
                y_range=[y_low, y_high, (y_high - y_low) / 4.0],
                x_length=3.8,
                y_length=2.7,
                tips=False,
                axis_config={"include_numbers": False, "stroke_width": 1.5},
            )
            labels = axes.get_axis_labels(Text(x_label, font_size=22), Text(y_label, font_size=22))
            axes_by_projection.append(axes)
            panels.add(VGroup(axes, labels))
        panels.arrange_in_grid(rows=2, cols=3, buff=(0.45, 0.4)).scale_to_fit_width(13.5)
        self.add(panels)

        clock = ValueTracker(float(time[0]))

        def point_now(projection_index: int, trajectory_index: int) -> np.ndarray:
            x_label, y_label = PROJECTIONS[projection_index]
            axes = axes_by_projection[projection_index]
            return axes.c2p(
                np.interp(clock.get_value(), time, variables[x_label][trajectory_index]),
                np.interp(clock.get_value(), time, variables[y_label][trajectory_index]),
            )

        for projection_index in range(len(PROJECTIONS)):
            for trajectory_index in range(state.shape[0]):
                color = COLORS[trajectory_index % len(COLORS)]
                self.add(
                    TracedPath(
                        lambda projection_index=projection_index, trajectory_index=trajectory_index: point_now(
                            projection_index, trajectory_index
                        ),
                        stroke_color=color,
                        stroke_width=0.9,
                    ),
                    always_redraw(
                        lambda projection_index=projection_index, trajectory_index=trajectory_index, color=color: Dot(
                            point_now(projection_index, trajectory_index), color=color, radius=0.035
                        )
                    ),
                )

        self.play(
            clock.animate.set_value(float(time[-1])),
            run_time=ANIMATION_SECONDS,
            rate_func=linear,
        )
