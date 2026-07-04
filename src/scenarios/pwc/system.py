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

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Return rebuttal-style inequality matrix ``A(x)``."""
        A, _, _ = self.get_coefficients(x)
        return torch.cat([-A, A], dim=1)

    def b(self, x: torch.Tensor) -> torch.Tensor:
        """Return rebuttal-style inequality bound ``b(x)``."""
        _, bl, bu = self.get_coefficients(x)
        return torch.cat([-bl, bu], dim=1)

    def get_lower_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Return lower bounds ``bl(x)``."""
        m = 2
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
        m1 = (x <= -1).expand(-1, m, -1)
        m2 = ((x > -1) & (x <= 0)).expand(-1, m, -1)
        m3 = ((x > 0) & (x <= 1)).expand(-1, m, -1)
        m4 = (x > 1).expand(-1, m, -1)
        area1 = m1 * torch.cat([bl11, bl12], dim=1)
        area2 = m2 * torch.cat([bl21, bl22], dim=1)
        area3 = m3 * torch.cat([bl31, bl33], dim=1)
        area4 = m4 * torch.cat([bl41, bl42], dim=1)
        return area1 + area2 + area3 + area4

    def get_upper_bound(self, x: torch.Tensor) -> torch.Tensor:
        """Return upper bounds ``bu(x)``."""
        m = 2
        bu11 = -3 * torch.sin(torch.pi * (x + 1) / 2) + 0.2
        bu12 = -3 * torch.sin(torch.pi * (x + 1) / 2) ** 3 + 1
        bu21 = torch.full_like(x, -2.0)
        bu22 = torch.full_like(x, 2.0)
        bu31 = 3 - 4 * (x - 0.5) ** 2
        bu32 = 3 - 4 * (x - 0.8) ** 2
        bu41 = torch.full_like(x, 2.0)
        bu42 = torch.full_like(x, 2.5)
        m1 = (x <= -1).expand(-1, m, -1)
        m2 = ((x > -1) & (x <= 0)).expand(-1, m, -1)
        m3 = ((x > 0) & (x <= 1)).expand(-1, m, -1)
        m4 = (x > 1).expand(-1, m, -1)
        area1 = m1 * torch.cat([bu11, bu12], dim=1)
        area2 = m2 * torch.cat([bu21, bu22], dim=1)
        area3 = m3 * torch.cat([bu31, bu32], dim=1)
        area4 = m4 * torch.cat([bu41, bu42], dim=1)
        return area1 + area2 + area3 + area4

    def get_coefficients(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coefficients for ``bl(x) <= A(x)y <= bu(x)``."""
        n_samples = x.shape[0]
        n_constraints = 2
        A = torch.ones(
            n_samples,
            n_constraints,
            self.y_dim,
            device=x.device,
            dtype=x.dtype,
        )
        return A, self.get_lower_bound(x), self.get_upper_bound(x)

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

    def get_constraint_violation_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Return rebuttal-style summed squared constraint violation."""
        A = self.A(x)
        b = self.b(x)
        residual = A @ y_pred - b
        violation = torch.clamp(residual, min=0.0)
        return (torch.norm(violation.squeeze(-1), dim=1) ** 2).sum()

    def get_ineq_err(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive inequality residual."""
        return torch.clamp(self.A(x) @ y_pred - self.b(x), min=0.0)

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
