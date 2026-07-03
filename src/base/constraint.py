"""Base interface for scenario-specific affine constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from box import Box


ConstraintMethod = Literal["CAffNet-FF", "CAffNet-TF", "HardNet"]


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

    def coefficients(
        self,
        x: torch.Tensor,
        method: ConstraintMethod,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return constraint coefficients for a method family.

        Prefer ``caffnet_coefficients`` or ``hardnet_coefficients`` when the
        caller already knows the method family. This dispatcher is useful in
        shared training/evaluation code.
        """
        if method in ("CAffNet-FF", "CAffNet-TF"):
            return self.caffnet_coefficients(x)
        if method == "HardNet":
            return self.hardnet_coefficients(x)
        raise ValueError(f"Unknown constraint method: {method}")
