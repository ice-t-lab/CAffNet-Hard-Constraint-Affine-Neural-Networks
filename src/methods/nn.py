"""Unconstrained neural network baseline."""

from __future__ import annotations

from typing import Literal

import torch
from box import Box

from src.base.net import BaseNet


Architecture = Literal["ff", "tf"]


class NN(BaseNet):
    """Plain network with no hard projection."""

    def __init__(
        self,
        cfg: Box,
        architecture: Architecture | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            name="NN",
            architecture=architecture,
        )
        self.network = self.build_network()

    def forward(
        self,
        x: torch.Tensor,
        f_nom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[N, y_dim, 1]`` network output."""
        y = self.forward_branch(self.network, x)
        if f_nom is not None:
            y = y + f_nom
        return y
