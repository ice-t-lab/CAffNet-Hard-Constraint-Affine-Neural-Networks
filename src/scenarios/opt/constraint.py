"""Optimization-solver affine constraints."""

from __future__ import annotations

import torch
from box import Box

from src.base.constraint import BaseConstraint

from .system import System


class Constraint(BaseConstraint):
    """Constraint coefficients for OPT CAffNet and HardNet methods."""

    def __init__(self, cfg: Box, system: System) -> None:
        super().__init__(cfg)
        self.system = system

    def caffnet_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.A(x)

    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.b(x)

    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        A, _, _ = self.system.get_coefficients(x)
        return A

    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.get_lower_bound(x)

    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.get_upper_bound(x)
