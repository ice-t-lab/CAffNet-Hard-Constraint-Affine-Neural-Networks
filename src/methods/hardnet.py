"""HardNet projection method."""

from __future__ import annotations

from typing import Literal

import torch
from box import Box

from src.base.constraint import BaseConstraint
from src.base.net import BaseNet
from src.utils.CAffine import hardnet_project


Architecture = Literal["ff", "tf"]


class HardNet(BaseNet):
    """Network followed by HardNet affine projection."""

    def __init__(
        self,
        cfg: Box,
        constraint: BaseConstraint,
        architecture: Architecture | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            name="HardNet",
            architecture=architecture,
        )
        self.constraint = constraint
        self.network = self.build_network()

    def apply_projection(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Project output to satisfy ``bl(x) <= A(x)y <= bu(x)``."""
        A, bl, bu = self.constraint.hardnet_coefficients(x)
        return hardnet_project(y=y, A=A, bl=bl, bu=bu)

    def forward(
        self,
        x: torch.Tensor,
        f_nom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[N, y_dim, 1]`` projected network output."""
        y = self.forward_branch(self.network, x)

        if f_nom is not None:
            y = y + f_nom

        return self.apply_projection(x, y)
