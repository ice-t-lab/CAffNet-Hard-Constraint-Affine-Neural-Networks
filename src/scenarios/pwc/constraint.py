"""Piecewise-constraint formulas."""

from __future__ import annotations

import torch
from box import Box

from src.base.constraint import BaseConstraint


class Constraint(BaseConstraint):
    """Affine constraints for the piecewise-constraint scenario."""

    def __init__(self, cfg: Box) -> None:
        super().__init__(cfg)
        self.n_constraints = 2

    def lower_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Return lower bounds ``bl(x)``."""
        bl11 = 5 * torch.sin(torch.pi * (x + 1) / 2) ** 2 - 3
        bl12 = 5 * torch.sin(torch.pi * (x + 1) / 2) ** 8 - 2
        bl21 = torch.full_like(x, -2.0)
        bl22 = torch.full_like(x, -3.0)
        bl31 = (4 - 9 * (x - 2 / 3) ** 2) * x - 2 - 0.5
        bl33 = (5 - 4 * (x - 2 / 3 + 1 / 2) ** 2) * x - 2 - 0.5
        bl41 = torch.where(x == 0, torch.zeros_like(x), 3 / x**3 - 2) - 0.5
        bl42 = (
            torch.where(x == 0, torch.zeros_like(x), 1.5 / x**3 - 0.5)
            - 0.5
            - 0.5
            - 5 / 18
        )

        m1, m2, m3, m4 = self.masks(x)
        area1 = m1 * torch.cat([bl11, bl12], dim=1)
        area2 = m2 * torch.cat([bl21, bl22], dim=1)
        area3 = m3 * torch.cat([bl31, bl33], dim=1)
        area4 = m4 * torch.cat([bl41, bl42], dim=1)
        return area1 + area2 + area3 + area4

    def upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Return upper bounds ``bu(x)``."""
        bu11 = -3 * torch.sin(torch.pi * (x + 1) / 2) + 0.2
        bu12 = -3 * torch.sin(torch.pi * (x + 1) / 2) ** 3 + 1
        bu21 = torch.full_like(x, -2.0)
        bu22 = torch.full_like(x, 2.0)
        bu31 = 3 - 4 * (x - 0.5) ** 2
        bu32 = 3 - 4 * (x - 0.8) ** 2
        bu41 = torch.full_like(x, 2.0)
        bu42 = torch.full_like(x, 2.5)

        m1, m2, m3, m4 = self.masks(x)
        area1 = m1 * torch.cat([bu11, bu12], dim=1)
        area2 = m2 * torch.cat([bu21, bu22], dim=1)
        area3 = m3 * torch.cat([bu31, bu32], dim=1)
        area4 = m4 * torch.cat([bu41, bu42], dim=1)
        return area1 + area2 + area3 + area4

    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``A(x)`` for ``bl(x) <= A(x)y <= bu(x)``."""
        return torch.ones(
            x.shape[0],
            self.n_constraints,
            self.cfg.system.y_dim,
            device=x.device,
            dtype=x.dtype,
        )

    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        """Return lower bounds for HardNet."""
        return self.lower_bound(x)

    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        """Return upper bounds for HardNet."""
        return self.upper_bound(x)

    def caffnet_A(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``A(x)`` for CAffNet constraints ``A(x)y <= b(x)``."""
        A = self.hardnet_A(x)
        return torch.cat([-A, A], dim=1)

    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``b(x)`` for CAffNet constraints ``A(x)y <= b(x)``."""
        return torch.cat([-self.lower_bound(x), self.upper_bound(x)], dim=1)

    def masks(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return piecewise-region masks expanded over constraints."""
        shape = (-1, self.n_constraints, -1)
        m1 = (x <= -1).expand(*shape)
        m2 = ((x > -1) & (x <= 0)).expand(*shape)
        m3 = ((x > 0) & (x <= 1)).expand(*shape)
        m4 = (x > 1).expand(*shape)
        return m1, m2, m3, m4
