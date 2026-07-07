"""Nominal PID controller for CBF rollouts."""

from __future__ import annotations

import torch
from box import Box
from torch import nn


class PIDController(nn.Module):
    """PID controller in the robot local frame."""

    def __init__(self, cfg: Box) -> None:
        super().__init__()
        self.dt = float(cfg.simulation.rollout.dt)
        self.k_px, self.k_py, self.k_ptheta = cfg.simulation.controller.k_p
        self.k_ix, self.k_iy, self.k_itheta = cfg.simulation.controller.k_i
        self.k_dx, self.k_dy, self.k_dtheta = cfg.simulation.controller.k_d
        self.integral: torch.Tensor | None = None
        self.prev_error: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        self.integral = torch.zeros(
            batch_size,
            3,
            1,
            device=device,
            dtype=torch.get_default_dtype(),
        )
        self.prev_error = torch.zeros_like(self.integral)

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        n_samples = error.shape[0]
        if self.integral is None or self.prev_error is None:
            self.reset(n_samples, error.device)
        if self.integral.dtype != error.dtype:
            self.integral = self.integral.to(dtype=error.dtype)
            self.prev_error = self.prev_error.to(dtype=error.dtype)

        derivative = (error - self.prev_error) / self.dt
        with torch.no_grad():
            self.integral = self.integral + error * self.dt
            self.prev_error = error.detach()

        x_e, y_e, theta_e = error[:, 0, :], error[:, 1, :], error[:, 2, :]
        int_x = self.integral[:, 0, :]
        int_y = self.integral[:, 1, :]
        int_theta = self.integral[:, 2, :]
        der_x = derivative[:, 0, :]
        der_y = derivative[:, 1, :]
        der_theta = derivative[:, 2, :]

        v = self.k_px * x_e + self.k_ix * int_x + self.k_dx * der_x
        omega = (
            self.k_py * y_e
            + self.k_iy * int_y
            + self.k_dy * der_y
            + self.k_ptheta * theta_e
            + self.k_itheta * int_theta
            + self.k_dtheta * der_theta
        )
        return torch.stack((v, omega), dim=1)
