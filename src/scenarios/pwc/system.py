"""Piecewise-constraint supervised system."""

from __future__ import annotations

import numpy as np
import torch
from box import Box

from src.base.system import BaseSystem


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class System(BaseSystem):
    """Target function for the PWC scenario."""

    def __init__(self, cfg: Box) -> None:
        self.x_dim = cfg.system.x_dim
        self.y_dim = cfg.system.y_dim
        super().__init__(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rebuttal piecewise target function."""
        area1 = (x <= -1) * (-5 * torch.sin(torch.pi * (x + 1) / 2) - 2)
        area2 = ((x > -1) & (x <= 0)) * -2
        area3 = ((x > 0) & (x <= 1)) * (2 - 9 * (x - 2 / 3) ** 2)
        area4 = (x > 1) * (3 / x**2 - 2)
        return area1 + area2 + area3 + area4


    def generate_x_train(self) -> torch.Tensor:
        """Generate uniform random training inputs."""
        low, high = self.cfg.simulation.data.train_range
        return self.uniform_x(self.cfg.simulation.data.n_train, low, high)

    def generate_x_eval(self) -> torch.Tensor:
        """Generate evenly spaced evaluation inputs."""
        low, high = self.cfg.simulation.data.eval_range
        return torch.linspace(
            low,
            high,
            self.cfg.simulation.data.n_eval,
            device=DEVICE,
            dtype=torch.get_default_dtype(),
        ).reshape(-1, self.x_dim, 1)

    def _generate_x(self, n_samples: int) -> torch.Tensor:
        """Fallback input generation."""
        low, high = self.cfg.simulation.data.train_range
        return self.uniform_x(n_samples, low, high)

    def get_main_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Return rebuttal-style summed squared prediction error."""
        error = y - y_pred
        return (torch.norm(error.squeeze(-1), dim=1) ** 2).sum()

    def uniform_x(
        self,
        n_samples: int,
        low: float,
        high: float,
    ) -> torch.Tensor:
        """Sample ``x`` uniformly in ``[low, high]``."""
        x = np.random.uniform(
            low,
            high,
            size=(n_samples, self.x_dim, 1),
        )
        return torch.tensor(x, device=DEVICE)
