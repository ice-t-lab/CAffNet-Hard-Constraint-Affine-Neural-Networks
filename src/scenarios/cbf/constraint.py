"""Affine CBF constraints for methods."""

from __future__ import annotations

import torch
from box import Box

from src.base.constraint import BaseConstraint

from .system import System


class Constraint(BaseConstraint):
    """Wrap system CBF coefficients for CAffNet and HardNet."""

    def __init__(self, cfg: Box, system: System) -> None:
        super().__init__(cfg)
        self.system = system

    def caffnet_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.A(x)

    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.b(x)

    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.get_coefficients(x)[0]

    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.get_coefficients(x)[1]

    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.get_coefficients(x)[2]
