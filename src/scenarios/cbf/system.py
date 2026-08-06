"""CBF scenario system and constraint geometry."""

from __future__ import annotations

import numpy as np
import torch
from box import Box

from src.base.system import BaseSystem

from .model import UnicycleModel
from .obstacle import Obstacle

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class System(BaseSystem):
    """Unicycle CBF system."""

    def __init__(self, cfg: Box) -> None:
        self.x_dim = int(cfg.system.x_dim)
        self.y_dim = int(cfg.system.y_dim)
        self.obs = [
            Obstacle(
                vertices=np.array(obs.vertices),
                center=np.array(obs.center),
                label=getattr(obs, "label", "obstacle"),
            )
            for obs in cfg.system.obstacles
        ]
        x_sample = (
            cfg.system.constraints.x_sample
            if "x_sample" in cfg.system.constraints
            else cfg.system.constraints.x
        )
        self.model = UnicycleModel(cfg)

        from .constraint import Constraint

        self.constraint = Constraint(cfg, self)
        self.x_sample_cons = self.constraint.box_polytope(x_sample)
        super().__init__(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.zero_y(x)

    def generate_y_train(self) -> None:
        return None

    def generate_y_eval(self) -> None:
        return None

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
                if torch.all(self.constraint.barrier(x_candidate)[0] >= 0):
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
