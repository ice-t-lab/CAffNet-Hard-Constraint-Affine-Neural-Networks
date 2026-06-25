"""Base interface for scenario-specific systems."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from box import Box


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

        self.logger.debug(
            "Initialized %s with x_dim=%s, y_dim=%s",
            self.__class__.__name__,
            cfg.system.x_dim,
            cfg.system.y_dim,
        )

    @abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Return the target output for ``x``.

        Supervised scenarios can use this to generate labels. Optimization and
        control scenarios may leave target generation to their simulation code.
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
