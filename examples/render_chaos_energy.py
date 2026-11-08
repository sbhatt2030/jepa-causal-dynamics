"""Render energy diagnostics for the chaos ensemble.

Run after ``generate_chaos_ensemble.py``:
    .venv\\Scripts\\manim -pql examples/render_chaos_energy.py ChaosEnergy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import BLUE, GREEN, ORANGE, PINK, PURPLE, RED, YELLOW, Axes, Dot, Scene, Text, TracedPath, ValueTracker, VGroup, always_redraw, linear

from sim.double_pendulum import DoublePendulumParameters, kinetic_energy, potential_energy, total_energy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "artifacts" / "chaos_damped_ensemble.npz"
COLORS = [BLUE, GREEN, YELLOW, ORANGE, RED, PINK, PURPLE]
ANIMATION_SECONDS = 60.0


def padded_range(values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    margin = max(0.08 * (high - low), 0.5)
    return low - margin, high + margin


class ChaosEnergy(Scene):
    def construct(self) -> None:
        data = np.load(TRAJECTORY_PATH)
        time = data["time"]
        state = data["state"]
        if time.ndim != 1 or state.ndim != 3 or state.shape[1:] != (time.size, 4):
            raise ValueError("expected time=(T,) and state=(ensemble_size, T, 4)")

        parameters = DoublePendulumParameters(
            mass_1=float(data["mass_1"]),
            mass_2=float(data["mass_2"]),
            length_1=float(data["length_1"]),
            length_2=float(data["length_2"]),
            gravity=float(data["gravity"]),
            damping_1=float(data["damping_1"]),
            damping_2=float(data["damping_2"]),
        )
        potential = potential_energy(state, parameters)
        kinetic = kinetic_energy(state, parameters)
        total = total_energy(state, parameters)
        time_by_trajectory = np.broadcast_to(time, state.shape[:2])
        panels_spec = (
            (potential, kinetic, "potential energy", "kinetic energy"),
            (time_by_trajectory, total, "time", "total energy"),
        )
        axes_list: list[Axes] = []
        panels = VGroup()
        for x_values, y_values, x_label, y_label in panels_spec:
            x_low, x_high = padded_range(x_values)
            y_low, y_high = padded_range(y_values)
            axes = Axes(
                x_range=[x_low, x_high, (x_high - x_low) / 4.0],
                y_range=[y_low, y_high, (y_high - y_low) / 4.0],
                x_length=5.8,
                y_length=5.0,
                tips=False,
                axis_config={"include_numbers": False, "stroke_width": 1.5},
            )
            axes_list.append(axes)
            panels.add(VGroup(axes, axes.get_axis_labels(Text(x_label, font_size=22), Text(y_label, font_size=22))))
        panels.arrange(buff=0.7).scale_to_fit_width(13.2)
        self.add(panels)

        clock = ValueTracker(float(time[0]))

        def point_now(panel_index: int, trajectory_index: int) -> np.ndarray:
            x_values, y_values, _, _ = panels_spec[panel_index]
            return axes_list[panel_index].c2p(
                np.interp(clock.get_value(), time, x_values[trajectory_index]),
                np.interp(clock.get_value(), time, y_values[trajectory_index]),
            )

        for panel_index in range(2):
            for trajectory_index in range(state.shape[0]):
                color = COLORS[trajectory_index % len(COLORS)]
                self.add(
                    TracedPath(
                        lambda panel_index=panel_index, trajectory_index=trajectory_index: point_now(panel_index, trajectory_index),
                        stroke_color=color,
                        stroke_width=1.25,
                    ),
                    always_redraw(
                        lambda panel_index=panel_index, trajectory_index=trajectory_index, color=color: Dot(
                            point_now(panel_index, trajectory_index), color=color, radius=0.05
                        )
                    ),
                )
        self.play(clock.animate.set_value(float(time[-1])), run_time=ANIMATION_SECONDS, rate_func=linear)
