"""CBF rollout training and evaluation."""

from __future__ import annotations

import os
import random
import time
from typing import Literal

import numpy as np
import torch

from src.base.simulation import BaseSimulation, Batch, LossTerms, Metrics, ResultData

from .controller import PIDController
from .system import System


class Simulation(BaseSimulation):
    """Train CBF controllers through multi-step rollouts."""

    system: System

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.system.model
        self.dt = float(self.cfg.simulation.rollout.dt)
        self.controller = PIDController(self.cfg).to(self.device)

    def before_train(self) -> None:
        seed = int(self.cfg.simulation.seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    def get_data(self, split: Literal["train", "eval"]) -> Batch:
        if split == "train":
            return {"x": self.system.x_train}
        return {"x": self.system.x_eval}

    def simulate(self, batch: Batch) -> LossTerms:
        rollout = self.rollout(
            batch["x"],
            horizon=float(self.cfg.simulation.rollout.t_train),
            store=False,
            measure_time=False,
        )
        main_loss = rollout["main_loss"]
        constraint_violation_loss = rollout["constraint_violation_loss"]
        total_loss = (
            main_loss
            + self.cfg.simulation.training.soft_weight * constraint_violation_loss
        )
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "constraint_violation_loss": constraint_violation_loss,
        }

    def evaluate(self, data: Batch | None = None) -> tuple[Metrics, ResultData]:
        x_0 = torch.tensor(
            self.cfg.simulation.rollout.x_0,
            device=self.device,
            dtype=torch.get_default_dtype(),
        ).view(1, self.system.x_dim, 1)
        self.net.eval()
        with torch.no_grad():
            rollout = self.rollout(
                x_0,
                horizon=float(self.cfg.simulation.rollout.t_eval),
                store=True,
                measure_time=True,
            )
        return self.metrics(rollout), self.result_data(rollout)

    def rollout(
        self,
        x: torch.Tensor,
        horizon: float,
        store: bool,
        measure_time: bool,
    ) -> dict[str, torch.Tensor | float]:
        n_steps = int(horizon / self.dt)
        n_samples, x_dim, _ = x.shape
        x_ref = torch.zeros(n_samples, x_dim, 1, device=x.device, dtype=x.dtype)
        x_k = x
        main_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        constraint_violation_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        test_time = 0.0
        self.controller.reset(batch_size=n_samples, device=x.device)

        Q, R, Qf = self.cost_matrices(x, n_samples)
        u_lb, u_ub = self.control_bounds(x, n_samples)
        x_store, u_nom_store, u_store = self.stores(n_steps, x) if store else (None, None, None)
        if store:
            x_store[0] = x_k[0].detach()

        for step in range(n_steps):
            x_k[:, 2, 0] = ((x_k[:, 2, 0] + torch.pi) % (2 * torch.pi)) - torch.pi
            error = self.system.global2local(x_ref, x_k)
            u_nom = torch.clamp(self.controller(error), u_lb, u_ub)

            if measure_time:
                start = time.perf_counter()
            u_k = self.net(x_k, u_nom)
            if measure_time:
                test_time += time.perf_counter() - start
            u_k = torch.clamp(u_k, u_lb, u_ub)

            rho = torch.sqrt(
                (x_ref[:, 0, 0] - x_k[:, 0, 0]) ** 2
                + (x_ref[:, 1, 0] - x_k[:, 1, 0]) ** 2
            )
            mask = (rho > 0.1).to(x_k.dtype).view(-1, 1, 1)
            u_k = u_k * mask
            x_k = x_k * mask
            main_loss = (
                main_loss
                + self.system.get_main_loss(x_ref, x_k, Q)
                + self.system.get_main_loss(u_k, u_nom, R)
            )
            constraint_violation_loss = (
                constraint_violation_loss
                + self.system.get_constraint_violation_loss(x_k, u_k)
            )

            if store:
                u_nom_store[step] = u_nom[0].detach()
                u_store[step] = u_k[0].detach()

            x_k = self.model.step(x_k, u_k, method="rk4")
            if store:
                x_store[step + 1] = x_k[0].detach()

        main_loss = main_loss + self.system.get_main_loss(x_ref, x_k, Qf)
        return {
            "main_loss": main_loss,
            "constraint_violation_loss": constraint_violation_loss,
            "total_loss": (
                main_loss
                + self.cfg.simulation.training.soft_weight * constraint_violation_loss
            ),
            "test_time": test_time,
            "x": x_store,
            "u_nom": u_nom_store,
            "u": u_store,
        }

    def metrics(self, rollout: dict[str, torch.Tensor | float]) -> Metrics:
        x = rollout["x"]
        u = rollout["u"]
        ineq_err = self.system.get_ineq_err(x[:-1], u)
        n_steps = u.shape[0]
        n_ineq = ineq_err.shape[1]
        return {
            "Cost": self.to_scalar(rollout["total_loss"]),
            "Max ineq viol.": self.to_scalar(
                torch.max(ineq_err, dim=1).values.sum() / n_steps
            ),
            "Mean ineq viol.": self.to_scalar(torch.mean(ineq_err, dim=1).sum() / n_steps),
            "Num ineq viol. (%)": self.to_scalar((ineq_err > 1e-6).sum() / n_steps / n_ineq),
            "Test Time (s)": float(rollout["test_time"]),
        }

    def result_data(self, rollout: dict[str, torch.Tensor | float]) -> ResultData:
        return {
            "x": rollout["x"].detach().cpu(),
            "u_nom": rollout["u_nom"].detach().cpu(),
            "u": rollout["u"].detach().cpu(),
        }

    def cost_matrices(
        self,
        like: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = self.diag(self.cfg.simulation.cost.Q, like).unsqueeze(0).expand(n_samples, -1, -1)
        R = self.diag(self.cfg.simulation.cost.R, like).unsqueeze(0).expand(n_samples, -1, -1)
        Qf = self.diag(self.cfg.simulation.cost.Qf, like).unsqueeze(0).expand(n_samples, -1, -1)
        return Q, R, Qf

    def control_bounds(
        self,
        like: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lb = torch.tensor(
            self.cfg.system.constraints.u.lb,
            device=like.device,
            dtype=like.dtype,
        ).view(1, -1, 1)
        ub = torch.tensor(
            self.cfg.system.constraints.u.ub,
            device=like.device,
            dtype=like.dtype,
        ).view(1, -1, 1)
        return lb.repeat(n_samples, 1, 1), ub.repeat(n_samples, 1, 1)

    def stores(
        self,
        n_steps: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_store = torch.zeros(
            n_steps + 1,
            self.system.x_dim,
            1,
            device=like.device,
            dtype=like.dtype,
        )
        u_nom_store = torch.zeros(
            n_steps,
            self.system.y_dim,
            1,
            device=like.device,
            dtype=like.dtype,
        )
        u_store = torch.zeros_like(u_nom_store)
        return x_store, u_nom_store, u_store

    @staticmethod
    def diag(values: list[float], like: torch.Tensor) -> torch.Tensor:
        return torch.diag(torch.tensor(values, device=like.device, dtype=like.dtype))
