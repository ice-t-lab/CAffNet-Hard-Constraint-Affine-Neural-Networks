"""Affine CBF constraints for methods."""

from __future__ import annotations

import numpy as np
import polytope
import torch
from box import Box

from src.base.constraint import BaseConstraint
from src.utils.smooth_cbf import smooth_union

from .obstacle import Obstacle
from .system import System


class Constraint(BaseConstraint):
    """Wrap system CBF coefficients for CAffNet and HardNet."""

    def __init__(self, cfg: Box, system: System) -> None:
        super().__init__(cfg)
        self.system = system
        self.kappa = float(cfg.system.cbf.smooth_factor)
        self.alpha = float(cfg.system.cbf.barrier_coefficient)
        self.state_cons = self.box_polytope(cfg.system.constraints.x)
        self.input_cons = self.box_polytope(cfg.system.constraints.u)

    def caffnet_A(self, x: torch.Tensor) -> torch.Tensor:
        _, _, Lgh_x = self.barrier(x)
        A_u = torch.tensor(
            self.input_cons.A,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([-Lgh_x, A_u], dim=1)

    def caffnet_b(self, x: torch.Tensor) -> torch.Tensor:
        h_x, Lfh_x, _ = self.barrier(x)
        b_u = torch.tensor(
            self.input_cons.b,
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1, 1)
        b_u = b_u.expand(x.shape[0], -1, -1)
        return torch.cat([Lfh_x + self.alpha * h_x, b_u], dim=1)

    def hardnet_A(self, x: torch.Tensor) -> torch.Tensor:
        return self.caffnet_A(x)

    def hardnet_bl(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(self.caffnet_b(x), -1e10)

    def hardnet_bu(self, x: torch.Tensor) -> torch.Tensor:
        return self.caffnet_b(x)

    def barrier(
        self,
        x: torch.Tensor,
        kappa: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return barrier values and their Lie derivatives."""
        kappa = self.kappa if kappa is None else kappa
        h_obs, Lfh_obs, Lgh_obs = self.obstacle_barriers(x, kappa)
        h_state, Lfh_state, Lgh_state = self.state_barriers(x)
        return (
            torch.cat((h_obs, h_state), dim=1),
            torch.cat((Lfh_obs, Lfh_state), dim=1),
            torch.cat((Lgh_obs, Lgh_state), dim=1),
        )

    def obstacle_barrier(
        self,
        x: torch.Tensor,
        obs: Obstacle,
        kappa: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kappa = self.kappa if kappa is None else kappa
        p = x[:, 0:2, :]
        A = torch.tensor(obs.P.A, device=x.device, dtype=x.dtype)
        A = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        b = torch.tensor(obs.P.b, device=x.device, dtype=x.dtype)
        b = b.view(1, -1, 1).expand(x.shape[0], -1, -1)
        h_i = A @ p - b
        Lfh_i = A @ self.system.model.f(x)[:, 0:2, :]
        Lgh_i = A @ self.system.model.g(x)[:, 0:2, :]
        return smooth_union(h_i, Lfh_i, Lgh_i, kappa)

    def obstacle_barriers(
        self,
        x: torch.Tensor,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = [self.obstacle_barrier(x, obs, kappa) for obs in self.system.obs]
        return tuple(torch.cat(parts, dim=1) for parts in zip(*values, strict=True))

    def state_barriers(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.tensor(self.state_cons.A, device=x.device, dtype=x.dtype)
        A = A.unsqueeze(0).expand(x.shape[0], -1, -1)
        b = torch.tensor(self.state_cons.b, device=x.device, dtype=x.dtype)
        b = b.view(1, -1, 1).expand(x.shape[0], -1, -1)
        h = b - A @ x
        return h, -A @ self.system.model.f(x), -A @ self.system.model.g(x)

    @staticmethod
    def box_polytope(bounds: Box) -> polytope.Polytope:
        return polytope.box2poly(
            np.array(list(zip(np.array(bounds.lb), np.array(bounds.ub))))
        )
