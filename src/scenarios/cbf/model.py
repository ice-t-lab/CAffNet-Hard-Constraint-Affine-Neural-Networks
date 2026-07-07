"""Unicycle dynamics for the CBF scenario."""

from __future__ import annotations

from typing import Literal

import torch
from box import Box


class UnicycleModel:
    """State is ``[px, py, theta]`` and control is ``[v, omega]``."""

    def __init__(self, cfg: Box) -> None:
        self.cfg = cfg
        self.dt = float(cfg.simulation.rollout.dt)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def g(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        g = torch.zeros(n_samples, 3, 2, dtype=x.dtype, device=x.device)
        g[:, 0, 0] = torch.cos(x[:, 2, 0])
        g[:, 1, 0] = torch.sin(x[:, 2, 0])
        g[:, 2, 1] = 1.0
        return g

    def step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        method: Literal["euler", "rk4"] = "rk4",
    ) -> torch.Tensor:
        if method == "euler":
            return self._step_euler(x, u)
        if method == "rk4":
            return self._step_rk4(x, u)
        raise ValueError(f"Unknown integration method: {method}")

    def _step_euler(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dxdt = self.f(x) + self.g(x) @ u
        return x + dxdt * self.dt

    def _step_rk4(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.f(x) + self.g(x) @ u
        k2 = self.f(x + 0.5 * dt * k1) + self.g(x + 0.5 * dt * k1) @ u
        k3 = self.f(x + 0.5 * dt * k2) + self.g(x + 0.5 * dt * k2) @ u
        k4 = self.f(x + dt * k3) + self.g(x + dt * k3) @ u
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
