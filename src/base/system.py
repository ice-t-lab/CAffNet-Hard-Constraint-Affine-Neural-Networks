"""Base interface for scenario-specific systems."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from box import Box

from .constraint import BaseConstraint


class BaseSystem(ABC):
    """Abstract parent class for all scenarios.

    A system owns scenario-level data generation and the target mapping
    ``y = f(x)`` when a scenario has supervised targets. Constraint formulas
    live in ``BaseConstraint`` so system logic and constraint logic stay
    modular.

    Expected tensor shapes:
        x: [N, cfg.system.x_dim, 1]
        y: [N, cfg.system.y_dim, 1]
    """

    def __init__(self, cfg: Box) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Build input samples once during initialization. Subclasses should set
        # any constants needed by _generate_x before calling super().__init__().
        self.x_train = self.generate_x_train()
        self.x_eval = self.generate_x_eval()
        self.y_train = self.generate_y_train()
        self.y_eval = self.generate_y_eval()

        self.logger.debug(
            "Initialized %s with x_dim=%s, y_dim=%s",
            self.__class__.__name__,
            cfg.system.x_dim,
            cfg.system.y_dim,
        )

    @abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Return the target output for ``x``.

        Subclasses with nonstandard labels can override ``generate_y_train``
        and ``generate_y_eval`` instead of using this default target mapping.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_x(self, n_samples: int) -> torch.Tensor:
        """Generate input samples.

        Subclasses decide the actual sampling distribution because each
        scenario has different state/input ranges and feasibility conditions.
        """
        raise NotImplementedError

    def generate_x_train(self) -> torch.Tensor:
        """Generate training inputs using ``cfg.simulation.data.n_train``."""
        n_train = self.cfg.simulation.data.n_train
        self.logger.debug("Generating %s training samples.", n_train)
        return self._generate_x(n_train)

    def generate_x_eval(self) -> torch.Tensor:
        """Generate evaluation inputs using ``cfg.simulation.data.n_eval``."""
        n_eval = self.cfg.simulation.data.n_eval
        self.logger.debug("Generating %s evaluation samples.", n_eval)
        return self._generate_x(n_eval)

    def generate_y_train(self) -> torch.Tensor:
        """Generate training targets from ``x_train``."""
        self.logger.debug("Generating training targets.")
        return self.f(self.x_train)

    def generate_y_eval(self) -> torch.Tensor:
        """Generate evaluation targets from ``x_eval``."""
        self.logger.debug("Generating evaluation targets.")
        return self.f(self.x_eval)

    def constraint_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        constraint: BaseConstraint | None,
        net_name: str,
    ) -> torch.Tensor | None:
        """Return positive residuals for affine constraints."""
        if constraint is None:
            return None

        if net_name == "HardNet":
            a, bl, bu = constraint.hardnet_coefficients(x)
            ay = torch.bmm(a.to(y.device), y)
            return torch.cat(
                [
                    torch.relu(bl.to(y.device) - ay),
                    torch.relu(ay - bu.to(y.device)),
                ],
                dim=1,
            )

        a, b = constraint.caffnet_coefficients(x)
        return torch.relu(torch.bmm(a.to(y.device), y) - b.to(y.device))
