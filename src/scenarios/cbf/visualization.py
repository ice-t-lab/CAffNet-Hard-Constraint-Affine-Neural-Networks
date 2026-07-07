"""CBF figures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box
from matplotlib.ticker import FormatStrFormatter

from src.base.visualization import BaseVisualization

from .system import DEVICE, System


class Visualization(BaseVisualization):
    """Trajectory and control plots for CBF rollouts."""

    def __init__(self, cfg: Box, system: System) -> None:
        super().__init__(cfg, system)

    def plot_problem(self, show_plot: bool = True) -> tuple[plt.Figure, plt.Axes]:
        cfg = self.cfg.visualization.problem
        fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)

        for obs in self.system.obs:
            obs.P.plot(ax, color="lightblue", edgecolor="black", alpha=0.8, linewidth=1)
            ax.text(
                obs.P.chebXc[0],
                obs.P.chebXc[1],
                rf"$\mathcal{{{obs.label}}}$",
                fontsize=cfg.obstacle_label_fontsize,
                ha="center",
                va="center",
            )
            self.plot_obstacle_boundary(ax, obs)

        ax.add_patch(plt.Circle((0, 0), 0.1, color="green", alpha=0.8, label="Goal Region"))
        ax.plot([], [], "go", label="$x_0$")
        ax.plot([], [], "ks", label="$x_N$")
        self._format_axis(ax, cfg)
        ax.legend()
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig, ax

    def plot_obstacle_boundary(self, ax: plt.Axes, obs: Any) -> None:
        constraints = self.cfg.system.constraints.x
        px = np.linspace(constraints.lb[0], constraints.ub[0], 400)
        py = np.linspace(constraints.lb[1], constraints.ub[1], 400)
        PX, PY = np.meshgrid(px, py)
        flat = np.vstack((PX.flatten(), PY.flatten(), np.zeros(PX.size))).T
        X = torch.tensor(flat, dtype=torch.get_default_dtype(), device=DEVICE).unsqueeze(-1)
        with torch.no_grad():
            H, _, _ = self.system._h_single_obs(X, obs, self.system.kappa)
        H = H.squeeze().cpu().numpy().reshape(PX.shape)
        ax.contour(PX, PY, H, levels=[0], colors="red", linewidths=2)

    def plot_result(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        x: torch.Tensor,
        method: str | None = None,
        show_plot: bool = True,
        **style: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        cfg = self.cfg.visualization.result
        plot_style = self._method_style(cfg, method)
        plot_style.update(style)
        x_np = self._to_numpy(x)

        ax.plot(
            x_np[:, 0, 0],
            x_np[:, 1, 0],
            c=plot_style.color,
            linestyle=plot_style.linestyle,
            linewidth=plot_style.linewidth,
            label=plot_style.label,
            alpha=plot_style.alpha,
        )
        ax.plot(x_np[0, 0, 0], x_np[0, 1, 0], "go")
        ax.plot(x_np[-1, 0, 0], x_np[-1, 1, 0], "ks")
        self._legend(ax, cfg)
        fig.canvas.draw()
        if show_plot:
            plt.show()
        return fig, ax

    def plot_controls(
        self,
        method_data: dict[str, dict[str, torch.Tensor]],
        show_plot: bool = True,
    ) -> tuple[plt.Figure, np.ndarray]:
        cfg = self.cfg.visualization.control
        fig, ax = plt.subplots(2, 1, figsize=cfg.figsize, dpi=cfg.dpi, sharex=True)
        dt = float(self.cfg.simulation.rollout.dt)
        nominal_plotted = False

        for method, data in method_data.items():
            u_nom = data["u_nom"]
            u = data["u"]
            x = data["x"]
            t = torch.arange(0, u.shape[0], dtype=u.dtype) * dt
            t_np = self._to_numpy(t)
            style = self._method_style(cfg, method)

            if not nominal_plotted:
                ax[0].plot(
                    t_np,
                    self._to_numpy(u_nom[:, 0, 0]),
                    c="blue",
                    linestyle="--",
                    linewidth=2,
                    label="$u_{nom}(x)$",
                )
                ax[1].plot(
                    t_np,
                    self._to_numpy(u_nom[:, 1, 0]),
                    c="blue",
                    linestyle="--",
                    linewidth=2,
                    label="$u_{nom}(x)$",
                )
                nominal_plotted = True

            self.plot_control_constraints(ax, t, x[:-1])
            ax[0].plot(
                t_np,
                self._to_numpy(u[:, 0, 0]),
                c=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                alpha=style.alpha,
                label=style.label,
            )
            ax[1].plot(
                t_np,
                self._to_numpy(u[:, 1, 0]),
                c=style.color,
                linestyle=style.linestyle,
                linewidth=style.linewidth,
                alpha=style.alpha,
                label=style.label,
            )

        self.format_control_axes(ax)
        if show_plot:
            plt.show()
        return fig, ax

    def plot_control_constraints(
        self,
        ax: np.ndarray,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
        A = self.system.A(x).detach().cpu().numpy()
        b = self.system.b(x).detach().cpu().numpy()
        t_np = self._to_numpy(t)
        fill_min, fill_max = -1000, 1000

        for index in range(A.shape[1]):
            Ai = A[:, index, :]
            bi = b[:, index, 0]
            if np.allclose(Ai[:, 1], 0, atol=1e-6):
                a1 = Ai[:, 0]
                bound = bi / a1
                ax[0].plot(t_np, np.where(a1 > 0, bound, np.nan), "r--", lw=1.0)
                ax[0].plot(t_np, np.where(a1 < 0, bound, np.nan), "r--", lw=1.0)
                ax[0].fill_between(t_np, bound, fill_max, where=a1 > 0, color="red", alpha=0.08)
                ax[0].fill_between(t_np, fill_min, bound, where=a1 < 0, color="red", alpha=0.08)
            elif np.allclose(Ai[:, 0], 0, atol=1e-6):
                a2 = Ai[:, 1]
                bound = bi / a2
                ax[1].plot(t_np, np.where(a2 > 0, bound, np.nan), "r--", lw=1.0)
                ax[1].plot(t_np, np.where(a2 < 0, bound, np.nan), "r--", lw=1.0)
                ax[1].fill_between(t_np, bound, fill_max, where=a2 > 0, color="red", alpha=0.08)
                ax[1].fill_between(t_np, fill_min, bound, where=a2 < 0, color="red", alpha=0.08)

    def format_control_axes(self, ax: np.ndarray) -> None:
        cfg = self.cfg.visualization.control
        u_bounds = self.cfg.system.constraints.u
        ax[0].set_ylabel(r"linear velocity, $v$ ($m/s$)")
        ax[1].set_ylabel(r"angular velocity, $\omega$ ($rad/s$)")
        ax[1].set_xlabel(r"time, $t$ ($s$)")
        ax[0].set_ylim([u_bounds.lb[0] - 0.5, u_bounds.ub[0] + 0.5])
        ax[1].set_ylim([u_bounds.lb[1] - 0.5, u_bounds.ub[1] + 0.5])

        for axis in ax:
            axis.grid(False)
            axis.yaxis.set_major_formatter(FormatStrFormatter(cfg.y_tick_format))
            axis.legend(
                loc=cfg.legend_loc,
                ncol=cfg.legend_ncol,
                frameon=cfg.legend_frameon,
            )
        self._apply_axis_fontsize(ax, cfg.label_fontsize, cfg.tick_fontsize)
        plt.tight_layout()
