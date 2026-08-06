"""Optimization-solver scenario system."""

from __future__ import annotations

import time

import numpy as np
import torch
from box import Box

from src.base.system import BaseSystem


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class System(BaseSystem):
    """Quadratic objective with affine equality and inequality constraints."""

    def __init__(self, cfg: Box) -> None:
        self.x_dim = int(cfg.system.x_dim)
        self.y_dim = int(cfg.system.y_dim)
        self.num_ineq = int(cfg.system.num_ineq)
        self.num_eq = int(cfg.system.num_eq)

        np.random.seed(int(cfg.simulation.seed))
        self.Q = np.diag(np.random.random(self.y_dim))
        self.p = np.random.random(self.y_dim)
        self.G = np.random.normal(loc=0.0, scale=1.0, size=(self.num_ineq, self.y_dim))
        self.C = np.random.normal(loc=0.0, scale=1.0, size=(self.num_eq, self.y_dim))

        C_dagger = torch.linalg.pinv(torch.tensor(self.C))
        self.h = torch.sum(torch.abs(torch.tensor(self.G) @ C_dagger), dim=1).numpy()

        super().__init__(cfg)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Solve the constrained optimization problem with GEKKO/IPOPT."""
        from gekko import GEKKO

        tol = 1e-4
        x_np = x.detach().cpu().numpy().reshape(x.shape[0], -1)
        y_store = []

        for xi in x_np:
            y0, *_ = np.linalg.lstsq(self.C, xi)

            model = GEKKO(remote=False)
            model.options.SOLVER = 3
            model.options.DIAGLEVEL = 1
            model.options.OTOL = tol
            model.options.RTOL = tol

            y = model.Array(model.Var, y0.shape, lb=-1e5, ub=1e5)
            for j, value in enumerate(y0):
                y[j].value = value

            for row, bound in zip(self.G, self.h, strict=True):
                model.Equation(np.dot(row, y) <= bound)
            for j, row in enumerate(self.C):
                model.Equation(np.dot(row, y) == xi[j])

            nonconvex = sum(self.p[i] * model.sin(y[i]) for i in range(len(y)))
            model.Minimize(model.sum(0.5 * (y * np.matmul(self.Q, y))) + nonconvex)
            model.solve(disp=False)
            y_store.append([yi.value[0] for yi in y])

        return torch.tensor(y_store, device=x.device, dtype=x.dtype).unsqueeze(-1)

    def solve_ipopt(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Solve all samples with IPOPT and return predictions plus total time."""
        start = time.perf_counter()
        y = self.f(x)
        return y, time.perf_counter() - start

    def generate_y_train(self) -> torch.Tensor:
        """Use dummy labels; OPT training optimizes the objective directly."""
        return self.zero_y(self.x_train)

    def generate_y_eval(self) -> torch.Tensor:
        """Use dummy labels; OPT evaluation uses objective and constraints."""
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
        low, high = self.cfg.simulation.data.train_range
        return self.uniform_x(self.cfg.simulation.data.n_train, low, high)

    def generate_x_eval(self) -> torch.Tensor:
        low, high = self.cfg.simulation.data.eval_range
        n_validation = int(getattr(self.cfg.simulation.data, "n_validation", 10))
        self.uniform_x(n_validation, low, high)
        return self.uniform_x(self.cfg.simulation.data.n_eval, low, high)

    def _generate_x(self, n_samples: int) -> torch.Tensor:
        low, high = self.cfg.simulation.data.train_range
        return self.uniform_x(n_samples, low, high)

    def uniform_x(self, n_samples: int, low: float, high: float) -> torch.Tensor:
        x = np.random.uniform(low, high, size=(n_samples, self.x_dim, 1))
        return torch.tensor(x, device=DEVICE, dtype=torch.get_default_dtype())

    def get_main_loss(
        self,
        y: torch.Tensor | None,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        n_samples = y_pred.shape[0]
        Q = self.Q_tensor(y_pred).unsqueeze(0).expand(n_samples, -1, -1)
        p = self.p_tensor(y_pred).unsqueeze(0).expand(n_samples, -1, -1)
        y_pred_T = y_pred.transpose(1, 2)
        loss = 0.5 * (y_pred_T @ Q @ y_pred) + (p.transpose(1, 2) @ torch.sin(y_pred))
        return loss.sum()

    def Q_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.Q, device=like.device, dtype=like.dtype)

    def p_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.p, device=like.device, dtype=like.dtype).unsqueeze(-1)

    def G_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.G, device=like.device, dtype=like.dtype)

    def C_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.C, device=like.device, dtype=like.dtype)

    def h_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.h, device=like.device, dtype=like.dtype).unsqueeze(-1)
