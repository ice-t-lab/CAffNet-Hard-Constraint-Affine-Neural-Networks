"""Base interface for scenario-specific affine constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from box import Box


class BaseConstraint(ABC):
    """Abstract interface for constraints used by CAffNet and HardNet.

    Scenario classes should implement the actual formulas. This base class only
    standardizes names and return shapes so method code can ask for constraints
    without knowing which scenario produced them.

    Expected tensor shapes:
        x:  [N, x_dim, 1]
        y:  [N, y_dim, 1]
        A:  [N, n_constraints, y_dim]
        b:  [N, n_constraints, 1]
        bl: [N, n_constraints, 1]
        bu: [N, n_constraints, 1]
    """

    def __init__(self, cfg: Box) -> None:
        self.cfg = cfg

    @abstractmethod
    def caffnet_A(self, x: torch.Tensor) -> torch.Tensor:
        """Return A(x) for CAffNet constraints A(x)y <= b(x)."""
        raise NotImplementedError

    @abstractmethod
    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        """Return b(x) for CAffNet constraints A(x)y <= b(x)."""
        raise NotImplementedError

    def caffnet_coefficients(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full CAffNet constraint tuple: A(x), b(x)."""
        return self.caffnet_A(x), self.caffnet_b(x)

    @abstractmethod
    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        """Return A(x) for HardNet constraints bl(x) <= A(x)y <= bu(x)."""
        raise NotImplementedError

    @abstractmethod
    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        """Return bl(x) for HardNet constraints bl(x) <= A(x)y <= bu(x)."""
        raise NotImplementedError

    @abstractmethod
    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        """Return bu(x) for HardNet constraints bl(x) <= A(x)y <= bu(x)."""
        raise NotImplementedError

    def hardnet_coefficients(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the full HardNet constraint tuple: A(x), bl(x), bu(x)."""
        return self.hardnet_A(x), self.hardnet_bl(x), self.hardnet_bu(x)

    def violation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the summed squared violation of the CAffNet inequalities."""
        residual = self.caffnet_A(x) @ y - self.caffnet_b(x)
        violation = torch.clamp(residual, min=0.0)
        return (torch.norm(violation.squeeze(-1), dim=1) ** 2).sum()

    def ineq_err(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return positive CAffNet inequality residuals."""
        return torch.clamp(self.caffnet_A(x) @ y - self.caffnet_b(x), min=0.0)
