"""CBF scenario system and constraint geometry."""

from __future__ import annotations

import numpy as np
import polytope
import torch
from box import Box

from src.base.system import BaseSystem
from src.utils.smooth_cbf import smooth_union

from .model import UnicycleModel
from .obstacle import Obstacle


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class System(BaseSystem):
    """Unicycle CBF system."""

    def __init__(self, cfg: Box) -> None:
        self.x_dim = int(cfg.system.x_dim)
        self.y_dim = int(cfg.system.y_dim)
        self.kappa = float(cfg.system.cbf.smooth_factor)
        self.alpha = float(cfg.system.cbf.barrier_coefficient)
        self.obs = [
            Obstacle(
                vertices=np.array(obs.vertices),
                center=np.array(obs.center),
                label=getattr(obs, "label", "obstacle"),
            )
            for obs in cfg.system.obstacles
        ]
        self.state_cons = self.box_polytope(cfg.system.constraints.x)
        self.input_cons = self.box_polytope(cfg.system.constraints.u)
        x_sample = (
            cfg.system.constraints.x_sample
            if "x_sample" in cfg.system.constraints
            else cfg.system.constraints.x
        )
        self.x_sample_cons = self.box_polytope(x_sample)
        self.model = UnicycleModel(cfg)
        super().__init__(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.zero_y(x)

    def generate_y_train(self) -> torch.Tensor:
        return self.zero_y(self.x_train)

    def generate_y_eval(self) -> torch.Tensor:
        return self.zero_y(self.x_eval)

    def zero_y(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0],
            self.y_dim,
            1,
            device=x.device,
            dtype=x.dtype,
        )

    def generate_x_train(self) -> torch.Tensor:
        # Rebuttal CBF generated data once in BaseSystem, then regenerated it.
        self._generate_x(self.cfg.simulation.data.n_train)
        self._generate_x(self.cfg.simulation.data.n_validation)
        self._generate_x(self.cfg.simulation.data.n_eval)
        return self._generate_x(self.cfg.simulation.data.n_train)

    def generate_x_eval(self) -> torch.Tensor:
        self._generate_x(self.cfg.simulation.data.n_validation)
        return self._generate_x(self.cfg.simulation.data.n_eval)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        _, _, Lgh_x = self.h(x, self.kappa)
        A_u = self.tensor(self.input_cons.A, x).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([-Lgh_x, A_u], dim=1)

    def b(self, x: torch.Tensor) -> torch.Tensor:
        h_x, Lfh_x, _ = self.h(x, self.kappa)
        b_u = self.tensor(self.input_cons.b, x).unsqueeze(-1)
        b_u = b_u.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([Lfh_x + self.alpha * h_x, b_u], dim=1)

    def get_coefficients(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.A(x)
        bu = self.b(x)
        bl = torch.zeros_like(bu) - 1e10
        return A, bl, bu

    def h(
        self,
        x: torch.Tensor,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_obs, Lfh_obs, Lgh_obs = self._h_obs(x, kappa)
        h_state, Lfh_state, Lgh_state = self._h_state(x)
        return (
            torch.cat((h_obs, h_state), dim=1),
            torch.cat((Lfh_obs, Lfh_state), dim=1),
            torch.cat((Lgh_obs, Lgh_state), dim=1),
        )

    def _h_single_obs(
        self,
        x: torch.Tensor,
        obs: Obstacle,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = x[:, 0:2, :]
        A = self.tensor(obs.P.A, x).unsqueeze(0).expand(x.shape[0], -1, -1)
        b = self.tensor(obs.P.b, x).unsqueeze(-1)
        b = b.unsqueeze(0).expand(x.shape[0], -1, -1)
        h_i = A @ p - b
        Lfh_i = A @ self.model.f(x)[:, 0:2, :]
        Lgh_i = A @ self.model.g(x)[:, 0:2, :]
        return smooth_union(h_i, Lfh_i, Lgh_i, kappa)

    def _h_obs(
        self,
        x: torch.Tensor,
        kappa: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_list, Lfh_list, Lgh_list = [], [], []
        for obs in self.obs:
            h_i, Lfh_i, Lgh_i = self._h_single_obs(x, obs, kappa)
            h_list.append(h_i)
            Lfh_list.append(Lfh_i)
            Lgh_list.append(Lgh_i)
        return (
            torch.cat(h_list, dim=1),
            torch.cat(Lfh_list, dim=1),
            torch.cat(Lgh_list, dim=1),
        )

    def _h_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.tensor(self.state_cons.A, x).unsqueeze(0).expand(x.shape[0], -1, -1)
        b = self.tensor(self.state_cons.b, x).unsqueeze(-1)
        b = b.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = b - A @ x
        Lfh = -A @ self.model.f(x)
        Lgh = -A @ self.model.g(x)
        return h, Lfh, Lgh

    def _generate_x(self, n_samples: int) -> torch.Tensor:
        x = torch.zeros(
            n_samples,
            self.x_dim,
            1,
            device=DEVICE,
            dtype=torch.get_default_dtype(),
        )
        x_lb = torch.tensor(
            self.x_sample_cons.bounding_box[0],
            device=DEVICE,
            dtype=torch.get_default_dtype(),
        ).view(1, self.x_dim, 1)
        x_ub = torch.tensor(
            self.x_sample_cons.bounding_box[1],
            device=DEVICE,
            dtype=torch.get_default_dtype(),
        ).view(1, self.x_dim, 1)

        for index in range(n_samples):
            valid = False
            while not valid:
                prob = torch.rand(
                    1,
                    self.x_dim,
                    1,
                    device=DEVICE,
                    dtype=torch.get_default_dtype(),
                )
                x_candidate = x_lb * (1 - prob) + x_ub * prob
                px = x_candidate[0, 0, 0]
                py = x_candidate[0, 1, 0]
                x_candidate[0, 2, 0] = torch.atan2(-py, -px)
                if torch.all(self.h(x_candidate, self.kappa)[0] >= 0):
                    valid = True
                    x[index] = x_candidate
        return x

    def global2local(self, x_r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        px_r = x_r[:, 0, :]
        py_r = x_r[:, 1, :]
        theta_r = x_r[:, 2, :]
        px = x[:, 0, :]
        py = x[:, 1, :]
        theta = x[:, 2, :]

        dx = px_r - px
        dy = py_r - py
        dtheta = theta_r - theta
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_e = cos_theta * dx + sin_theta * dy
        y_e = -sin_theta * dx + cos_theta * dy
        return torch.stack((x_e, y_e, dtheta), dim=1)

    def get_main_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        error = torch.sqrt(Q) @ (y - y_pred)
        return (torch.norm(error.squeeze(-1), dim=1) ** 2).sum()

    def get_constraint_violation_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.A(x) @ y_pred - self.b(x)
        violation = torch.clamp(residual, min=0.0)
        return (torch.norm(violation.squeeze(-1), dim=1) ** 2).sum()

    def get_ineq_err(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(self.A(x) @ y_pred - self.b(x), min=0.0)

    @staticmethod
    def box_polytope(bounds: Box) -> polytope.Polytope:
        return polytope.box2poly(
            np.array(list(zip(np.array(bounds.lb), np.array(bounds.ub))))
        )

    @staticmethod
    def tensor(value: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(value, device=like.device, dtype=like.dtype)
