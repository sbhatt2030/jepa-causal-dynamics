"""Pure geometry adapter for Manim (or any other renderer)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bob_positions(state: NDArray[np.float64], length_1: float, length_2: float) -> NDArray[np.float64]:
    """Map state rows to ``(pivot, bob_1, bob_2)`` Cartesian positions.

    The y axis points upward.  This is deliberately renderer-independent: a
    Manim scene can call it frame by frame, while Matplotlib can plot it.
    """
    states = np.atleast_2d(np.asarray(state, dtype=np.float64))
    theta_1, theta_2 = states[:, 0], states[:, 1]
    pivot = np.zeros((states.shape[0], 2), dtype=np.float64)
    bob_1 = np.column_stack((length_1 * np.sin(theta_1), -length_1 * np.cos(theta_1)))
    bob_2 = bob_1 + np.column_stack((length_2 * np.sin(theta_2), -length_2 * np.cos(theta_2)))
    return np.stack((pivot, bob_1, bob_2), axis=1)
