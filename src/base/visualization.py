"""Base plotting utilities for scenarios."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box

from .system import BaseSystem


class BaseVisualization:
    """Default visualization for one-dimensional supervised scenarios."""

    def __init__(self, cfg: Box, system: BaseSystem) -> None:
        self.cfg = cfg
        self.system = system

    def plot_problem(self, show_plot: bool = True) -> tuple[plt.Figure, plt.Axes]:
        """Plot train/eval data for one-dimensional x and y."""
        cfg = self.cfg.visualization.problem
        x_train = self._to_numpy(self.system.x_train).squeeze()
        y_train = self._to_numpy(self.system.y_train).squeeze()
        x_eval = self._to_numpy(self.system.x_eval).squeeze()
        y_eval = self._to_numpy(self.system.y_eval).squeeze()

        if x_train.ndim != 1 or y_train.ndim != 1:
            raise NotImplementedError(
                "Default plot_problem only supports one-dimensional x and y."
            )

        fig, ax = plt.subplots(
            figsize=cfg.figsize,
            dpi=cfg.dpi,
        )
        ax.scatter(
            x_train,
            y_train,
            c="grey",
            s=50,
            alpha=0.3,
            label="Training Points",
        )
        ax.plot(
            x_eval,
            y_eval,
            c="black",
            alpha=0.3,
            linewidth=0.5,
            label="Ground Truth",
        )

        self._format_axis(ax, cfg)
        if "legend_fontsize" in cfg:
            ax.legend(fontsize=cfg.legend_fontsize)
        else:
            ax.legend()
        plt.tight_layout()
        if show_plot:
            plt.show()
        return fig, ax

    def plot_result(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str | None = None,
        show_plot: bool = True,
        **style: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Add a model result curve to an existing figure."""
        cfg = self.cfg.visualization.result
        plot_style = self._method_style(cfg, method)
        plot_style.update(style)

        ax.plot(
            self._to_numpy(x).squeeze(),
            self._to_numpy(y).squeeze(),
            c=plot_style.color,
            linestyle=plot_style.linestyle,
            linewidth=plot_style.linewidth,
            label=plot_style.label,
            alpha=plot_style.alpha,
        )

        self._legend(ax, cfg)
        fig.canvas.draw()
        if show_plot:
            plt.show()
        return fig, ax

    def _format_axis(self, ax: plt.Axes, cfg: Box) -> None:
        ax.set_xlim(cfg.xlim)
        ax.set_ylim(cfg.ylim)
        ax.set_xlabel(cfg.xlabel)
        ax.set_ylabel(cfg.ylabel)
        ax.grid(cfg.grid)
        if "aspect" in cfg:
            ax.set_aspect(cfg.aspect)
        self._apply_axis_fontsize(
            ax,
            cfg.label_fontsize,
            cfg.tick_fontsize,
        )

    @staticmethod
    def _apply_axis_fontsize(
        ax: plt.Axes | np.ndarray,
        label_fontsize: float | None = None,
        tick_fontsize: float | None = None,
    ) -> None:
        axes = np.atleast_1d(ax)
        for axis in axes:
            if label_fontsize is not None:
                axis.xaxis.label.set_size(label_fontsize)
                axis.yaxis.label.set_size(label_fontsize)
            if tick_fontsize is not None:
                axis.tick_params(axis="both", labelsize=tick_fontsize)

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _method_style(
        self,
        cfg: Box,
        method: str | None,
    ) -> Box:
        style = Box(
            {
                "color": "red",
                "linestyle": "-",
                "linewidth": cfg.linewidth,
                "label": method or "Model",
                "alpha": cfg.alpha,
            }
        )
        if method is None:
            return style

        method_styles = cfg.method_styles
        for method_key in (method, method.replace("-", "_")):
            if method_key in method_styles:
                style.update(method_styles[method_key])
                return style
        return style

    def _legend(self, ax: plt.Axes, cfg: Box) -> None:
        legend_cfg = {}
        if "legend_loc" in cfg:
            legend_cfg["loc"] = cfg.legend_loc
        if "legend_bbox_to_anchor" in cfg:
            legend_cfg["bbox_to_anchor"] = cfg.legend_bbox_to_anchor
        if "legend_ncol" in cfg:
            legend_cfg["ncol"] = cfg.legend_ncol
        if "legend_frameon" in cfg:
            legend_cfg["frameon"] = cfg.legend_frameon
        if "legend_fontsize" in cfg:
            legend_cfg["fontsize"] = cfg.legend_fontsize
        ax.legend(**legend_cfg)
