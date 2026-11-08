"""Animate an ensemble of double pendulums and its aggregate energy budget.

The trajectories are recalculated every time Manim constructs this scene; no
``.npz`` input file is needed.  ``PENDULUM_COUNT`` independent double
pendulums share the same physical parameters but have different initial
states.  The chart displays the sum across all of them, with gravitational
potential shifted so that the downward-rest configuration is zero.

Render a one-minute movie with:
    .venv\\Scripts\\manim -pql examples/render_n_pendulum_energy.py NPendulumEnergy

For a final 60 fps render, replace ``-pql`` with ``-pqh -r 1920,1080 --fps 60``.
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    GREEN,
    ORIGIN,
    ORANGE,
    RED,
    UP,
    Axes,
    DecimalNumber,
    Dot,
    Line,
    Scene,
    Text,
    TracedPath,
    ValueTracker,
    VGroup,
    always_redraw,
    linear,
)

from sim.animation import bob_positions
from sim.double_pendulum import (
    DoublePendulumParameters,
    InputSchedules,
    IntegrationConfig,
    kinetic_energy,
    potential_energy,
    simulate_double_pendulum,
)

# These are deliberately simple module constants so a presentation can use a
# different ensemble size or physical duration without changing the scene.
PENDULUM_COUNT = 4
PHYSICAL_SECONDS = 60.0
ANIMATION_SECONDS = 60.0
INTEGRATION_DT = 0.005
OUTPUT_STRIDE = 4
PENDULUM_COLORS = (BLUE, GREEN, ORANGE, RED)


def simulate_ensemble(
    count: int = PENDULUM_COUNT,
    duration: float = PHYSICAL_SECONDS,
) -> tuple[np.ndarray, np.ndarray, DoublePendulumParameters]:
    """Return a uniformly sampled, unforced ensemble for the animation.

    The individual systems are conservative, so numerical drift in the gold
    total-energy trace should remain negligible.  Each row of ``state`` is a
    separate double pendulum and has shape ``(T, 4)``.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    if duration <= 0:
        raise ValueError("duration must be positive")

    parameters = DoublePendulumParameters()
    steps = round(duration / INTEGRATION_DT)
    config = IntegrationConfig(
        dt=INTEGRATION_DT,
        steps=steps,
        method="RK4",
        output_stride=OUTPUT_STRIDE,
    )
    # Asymmetric, distinct releases make the energy exchange and chaotic
    # motion visible while preserving an identical physical system.
    phases = np.linspace(-0.22, 0.22, count)
    initial_states = np.column_stack(
        (
            1.85 + phases,
            -1.25 - 0.6 * phases,
            0.16 * np.sin(4.0 * phases),
            0.12 * np.cos(3.0 * phases),
        )
    )
    trajectories = [
        simulate_double_pendulum(state, parameters, InputSchedules(), config)
        for state in initial_states
    ]
    return trajectories[0].time, np.stack([trajectory.state for trajectory in trajectories]), parameters


class NPendulumEnergy(Scene):
    """A one-minute synchronized motion and energy visualization."""

    def construct(self) -> None:
        time, state, parameters = simulate_ensemble()
        kinetic = kinetic_energy(state, parameters).sum(axis=0)
        potential = potential_energy(state, parameters)
        potential_minimum = -(
            (parameters.mass_1 + parameters.mass_2) * parameters.gravity * parameters.length_1
            + parameters.mass_2 * parameters.gravity * parameters.length_2
        )
        potential_above_rest = (potential - potential_minimum).sum(axis=0)
        total = kinetic + potential_above_rest

        clock = ValueTracker(float(time[0]))
        title = Text(
            f"Energy exchange in {PENDULUM_COUNT} double pendulums", font_size=32
        ).to_edge(UP)
        self.add(title)

        # The pendulums occupy a two-by-two grid on the left of the dashboard.
        pivots = np.array(((-4.7, 2.0, 0.0), (-2.6, 2.0, 0.0), (-4.7, -1.2, 0.0), (-2.6, -1.2, 0.0)))
        rows = int(np.ceil(PENDULUM_COUNT / 2))
        if rows > 2:
            pivots = np.array(
                [(-4.8 + 2.0 * (index % 2), 2.0 - 2.0 * (index // 2), 0.0) for index in range(PENDULUM_COUNT)]
            )

        def positions_now(index: int) -> np.ndarray:
            current = np.array(
                [np.interp(clock.get_value(), time, state[index, :, column]) for column in range(4)]
            )
            points = bob_positions(current, parameters.length_1, parameters.length_2)[0]
            return np.column_stack((points, np.zeros(3))) + pivots[index]

        for index in range(PENDULUM_COUNT):
            color = PENDULUM_COLORS[index % len(PENDULUM_COLORS)]
            self.add(
                Dot(pivots[index], color=color, radius=0.045),
                always_redraw(lambda index=index, color=color: Line(positions_now(index)[0], positions_now(index)[1], color=color, stroke_width=2)),
                always_redraw(lambda index=index, color=color: Dot(positions_now(index)[1], color=color, radius=0.07)),
                always_redraw(lambda index=index, color=color: Line(positions_now(index)[1], positions_now(index)[2], color=color, stroke_width=2)),
                always_redraw(lambda index=index, color=color: Dot(positions_now(index)[2], color=color, radius=0.085)),
            )

        y_max = 1.08 * float(max(kinetic.max(), potential_above_rest.max(), total.max()))
        axes = Axes(
            x_range=[0, PHYSICAL_SECONDS, 10],
            y_range=[0, y_max, max(1.0, round(y_max / 5.0))],
            x_length=6.2,
            y_length=5.4,
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": 1.5},
        ).shift(np.array((3.4, -0.5, 0.0)))
        labels = axes.get_axis_labels(Text("time (s)", font_size=20), Text("energy (J)", font_size=20))
        self.add(axes, labels)

        traces = ((kinetic, BLUE, "kinetic"), (potential_above_rest, GREEN, "potential"), (total, ORANGE, "total"))

        def point_now(values: np.ndarray) -> np.ndarray:
            now = clock.get_value()
            return axes.c2p(now, np.interp(now, time, values))

        legend = VGroup()
        for values, color, name in traces:
            swatch = Line(ORIGIN, 0.32 * np.array((1.0, 0.0, 0.0)), color=color, stroke_width=4)
            legend.add(VGroup(swatch, Text(name, font_size=20).next_to(swatch, direction=np.array((1.0, 0.0, 0.0)), buff=0.08)))
            self.add(
                always_redraw(lambda values=values, color=color: Dot(point_now(values), color=color, radius=0.05)),
                # A traced path is substantially cheaper than regenerating a
                # polyline every frame and keeps the plot visibly live.
                TracedPath(lambda values=values: point_now(values), stroke_color=color, stroke_width=2),
            )
        legend.arrange(direction=np.array((0.0, -1.0, 0.0)), aligned_edge=np.array((-1.0, 0.0, 0.0)), buff=0.13)
        legend.next_to(axes, direction=np.array((1.0, 0.0, 0.0)), buff=0.2).shift(np.array((-0.1, 1.75, 0.0)))
        self.add(legend)

        total_readout = always_redraw(
            lambda: VGroup(
                Text("total:", font_size=22, color=ORANGE),
                DecimalNumber(np.interp(clock.get_value(), time, total), num_decimal_places=2, font_size=22, color=ORANGE),
                Text("J", font_size=22, color=ORANGE),
            ).arrange(buff=0.08).next_to(axes, direction=np.array((0.0, -1.0, 0.0)), buff=0.25)
        )
        self.add(total_readout)
        self.play(clock.animate.set_value(float(time[-1])), run_time=ANIMATION_SECONDS, rate_func=linear)
