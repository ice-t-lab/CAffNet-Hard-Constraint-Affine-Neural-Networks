"""Polygonal obstacles for the CBF scenario."""

from __future__ import annotations

import numpy as np
import polytope


class Obstacle:
    """Obstacle stored as a polytope."""

    def __init__(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
        label: str = "obstacle",
    ) -> None:
        self.label = label
        self.P = polytope.qhull(vertices + center)
