"""PWC visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from box import Box

from src.base.visualization import BaseVisualization

from .constraint import Constraint
from .system import System


class Visualization(BaseVisualization):
    """Plot PWC samples, ground truth, and feasible region."""

    def __init__(self, cfg: Box, system: System, constraint: Constraint) -> None:
        super().__init__(cfg, system)
        self.constraint = constraint

    def plot_problem(self, show_plot: bool = True) -> tuple[plt.Figure, plt.Axes]:
        """Plot the PWC problem."""
        cfg = self.cfg.visualization.problem
        x_train = self.system.x_train
        y_train = self.system.y_train
        x_eval = self.system.x_eval
        y_eval = self.system.y_eval
        A, bl, bu = self.constraint.hardnet_coefficients(x_eval)

        fig, ax = plt.subplots(figsize=cfg.figsize, dpi=cfg.dpi)
        ax.scatter(
            x_train.squeeze().cpu(),
            y_train.squeeze().cpu(),
            c="grey",
            s=50,
            label="Training Points",
            alpha=0.3,
        )
        ax.plot(
            x_eval.squeeze().cpu(),
            y_eval.squeeze().cpu(),
            c="black",
            label="Ground Truth",
            alpha=0.3,
            linewidth=0.5,
        )

        for i in range(A.shape[1]):
            ax.plot(
                x_eval.squeeze().cpu(),
                bl[:, i].squeeze().cpu(),
                linestyle="--",
                linewidth=1,
                color="red",
            )
            ax.plot(
                x_eval.squeeze().cpu(),
                bu[:, i].squeeze().cpu(),
                linestyle="--",
                linewidth=1,
                color="red",
            )

        lower_bound = torch.max(bl[:, 0], bl[:, 1])
        upper_bound = torch.min(bu[:, 0], bu[:, 1])
        ax.fill_between(
            x_eval.squeeze().cpu(),
            lower_bound.squeeze().cpu(),
            upper_bound.squeeze().cpu(),
            color="green",
            alpha=0.1,
            label="Feasible Region",
        )

        self._format_axis(ax, cfg)
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig, ax
