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
        G = self.system.G_tensor(x)
        C = self.system.C_tensor(x)
        A = torch.cat([G, C, -C], dim=0)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)

    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        h = self.system.h_tensor(x).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([h, x, -x], dim=1)

    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.cat([self.system.G_tensor(x), self.system.C_tensor(x)], dim=0)
        return A.unsqueeze(0).expand(x.shape[0], -1, -1)

    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        lower_ineq = torch.full(
            (x.shape[0], self.system.num_ineq, 1),
            -1e10,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([lower_ineq, x], dim=1)

    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        h = self.system.h_tensor(x).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([h, x], dim=1)

    def ineq_err(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        G = self.system.G_tensor(y).unsqueeze(0).expand(y.shape[0], -1, -1)
        h = self.system.h_tensor(y).unsqueeze(0).expand(y.shape[0], -1, -1)
        return torch.clamp(G @ y - h, min=1e-6)

    def eq_err(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = self.system.C_tensor(y).unsqueeze(0).expand(y.shape[0], -1, -1)
        return torch.abs(C @ y - x)
