"""Saved-result loading, tables, and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from box import Box

from .visualization import BaseVisualization


Metrics = dict[str, Any]
ResultData = dict[str, Any]


class BaseResult:
    """Build result artifacts from saved data, without evaluating networks."""

    def __init__(
        self,
        cfg: Box,
        visualization: BaseVisualization,
    ) -> None:
        self.cfg = cfg
        self.result_dir = Path(cfg.result_dir)
        self.visualization = visualization
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        method: str,
        metrics: Metrics | pd.DataFrame | None = None,
        result_data: ResultData | None = None,
        loss_history: Iterable[Metrics] | pd.DataFrame | None = None,
    ) -> None:
        """Save all data needed by the result file."""
        if metrics is not None:
            self.save_metrics(method, metrics)
        if result_data is not None:
            self.save_result_data(method, result_data)
        if loss_history is not None:
            self.save_loss_history(method, loss_history)

    def save_metrics(self, method: str, metrics: Metrics | pd.DataFrame) -> None:
        """Save one-row evaluation metrics for a method."""
        self.method_dir(method).mkdir(parents=True, exist_ok=True)
        if isinstance(metrics, pd.DataFrame):
            table = metrics.copy()
        else:
            table = pd.DataFrame(
                [{key: self._to_scalar(value) for key, value in metrics.items()}]
            )

        if "Method" not in table:
            table.insert(0, "Method", method)
        table.to_csv(self.metrics_path(method), index=False)

    def save_result_data(self, method: str, result_data: ResultData) -> None:
        """Save tensors needed for plotting a method result."""
        self.method_dir(method).mkdir(parents=True, exist_ok=True)
        torch.save(result_data, self.result_path(method))

    def save_loss_history(
        self,
        method: str,
        loss_history: Iterable[Metrics] | pd.DataFrame,
    ) -> None:
        """Save training loss history for later plotting."""
        self.method_dir(method).mkdir(parents=True, exist_ok=True)
        if isinstance(loss_history, pd.DataFrame):
            loss = loss_history.copy()
        else:
            loss = pd.DataFrame(loss_history)
        loss.to_csv(self.loss_csv_path(method), index=False)
        torch.save(loss.to_dict(orient="list"), self.loss_pt_path(method))

    def table(
        self,
        methods: list[str] | None = None,
        filename: str = "evaluation.csv",
    ) -> pd.DataFrame:
        """Combine saved per-method metric CSVs."""
        tables = [pd.read_csv(path) for path in self.metrics_paths(methods)]
        if not tables:
            raise FileNotFoundError(
                f"No evaluation CSV files found in {self.result_dir}"
            )

        table = pd.concat(tables, ignore_index=True)
        if methods is not None and "Method" in table:
            order = {method: i for i, method in enumerate(methods)}
            table = table.sort_values(
                by="Method",
                key=lambda col: col.map(order).fillna(len(order)),
            )
        table.to_csv(self.result_dir / filename, index=False)
        return table

    def latex_table(
        self,
        methods: list[str] | None = None,
        filename: str = "evaluation.tex",
    ) -> str:
        """Create a rebuttal-style LaTeX table from saved metrics."""
        methods = self.metric_methods() if methods is None else methods
        df_list = [
            self._metrics_for_latex(self.load_metrics(method))
            for method in methods
        ]
        table = combineDfs(df_list, methods)
        latex = df2latex(table)
        (self.result_dir / filename).write_text(latex, encoding="utf-8")
        return latex

    def plot_results(
        self,
        methods: list[str] | None = None,
        filename: str = "result.png",
        show_plot: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot saved method results on the problem figure."""
        methods = self.result_methods() if methods is None else methods
        if not methods:
            raise FileNotFoundError(f"No result data files found in {self.result_dir}")
        fig, ax = self.visualization.plot_problem(show_plot=False)

        for method in methods:
            data = self.load_result_data(method)
            fig, ax = self.plot_result_data(fig, ax, method, data)

        fig.savefig(self.result_dir / filename)
        if show_plot:
            plt.show()
        return fig, ax

    def plot_loss(
        self,
        methods: list[str] | None = None,
        filename: str = "loss.png",
        show_plot: bool = False,
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot saved loss histories."""
        methods = self.loss_methods() if methods is None else methods
        losses = {method: self.load_loss_history(method) for method in methods}
        if not losses:
            raise FileNotFoundError(f"No loss files found in {self.result_dir}")

        first = next(iter(losses.values()))
        terms = [
            term
            for term in ("main_loss", "constraint_violation_loss", "total_loss")
            if term in first
        ]
        if not terms:
            raise ValueError("Saved loss history has no plottable loss columns.")

        cfg = self.cfg.visualization.loss_history
        fig, axs = plt.subplots(
            len(terms),
            1,
            figsize=cfg.figsize,
            dpi=cfg.dpi,
            sharex=True,
        )
        axs = [axs] if len(terms) == 1 else list(axs)

        for method, loss in losses.items():
            if "first_n_epochs" in cfg:
                loss = loss.head(cfg.first_n_epochs)
            epoch = loss["epoch"] if "epoch" in loss else range(1, len(loss) + 1)
            style = self.method_style(cfg, method)
            for ax, term in zip(axs, terms, strict=True):
                ax.plot(
                    epoch,
                    loss[term],
                    c=style.color,
                    linestyle=style.linestyle,
                    linewidth=style.linewidth,
                    alpha=style.alpha,
                    label=style.label,
                )

        for ax, term in zip(axs, terms, strict=True):
            ax.set_ylabel(term)
            ax.grid(True)
            if "ylim" in cfg:
                ax.set_ylim(cfg.ylim)
            ax.legend(fontsize=cfg.legend_fontsize)
            self.visualization._apply_axis_fontsize(
                ax,
                cfg.label_fontsize,
                cfg.tick_fontsize,
            )
        axs[-1].set_xlabel("epoch")

        plt.tight_layout()
        fig.savefig(self.result_dir / filename)
        if show_plot:
            plt.show()
        return fig, axs

    def plot_result_data(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        method: str,
        data: ResultData,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot one saved result. Override for scenario-specific payloads."""
        return self.visualization.plot_result(
            fig,
            ax,
            method=method,
            show_plot=False,
            **data,
        )

    def load_result_data(self, method: str) -> ResultData:
        """Load saved plot data for a method."""
        return torch.load(self.result_path(method), map_location="cpu")

    def load_metrics(self, method: str) -> pd.DataFrame:
        """Load saved evaluation metrics for a method."""
        return pd.read_csv(self.metrics_path(method))

    def load_loss_history(self, method: str) -> pd.DataFrame:
        """Load saved loss history for a method."""
        path = self.loss_csv_path(method)
        if path.exists():
            return pd.read_csv(path)
        data = torch.load(self.loss_pt_path(method), map_location="cpu")
        return pd.DataFrame(data)

    def metrics_paths(self, methods: list[str] | None = None) -> list[Path]:
        """Return per-method evaluation CSV paths."""
        if methods is not None:
            return [self.metrics_path(method) for method in methods]
        return sorted(self.result_dir.glob("*/evaluation.csv"))

    def metric_methods(self) -> list[str]:
        """Return methods with saved metric data."""
        return [
            path.parent.name
            for path in sorted(self.result_dir.glob("*/evaluation.csv"))
        ]

    def result_methods(self) -> list[str]:
        """Return methods with saved result data."""
        return [
            path.parent.name
            for path in sorted(self.result_dir.glob("*/result.pt"))
        ]

    def loss_methods(self) -> list[str]:
        """Return methods with saved loss data."""
        return [
            path.parent.name
            for path in sorted(self.result_dir.glob("*/loss.csv"))
        ]

    def method_dir(self, method: str) -> Path:
        return self.result_dir / method

    def metrics_path(self, method: str) -> Path:
        return self.method_dir(method) / "evaluation.csv"

    def result_path(self, method: str) -> Path:
        return self.method_dir(method) / "result.pt"

    def loss_csv_path(self, method: str) -> Path:
        return self.method_dir(method) / "loss.csv"

    def loss_pt_path(self, method: str) -> Path:
        return self.method_dir(method) / "loss.pt"

    def method_style(self, cfg: Box, method: str) -> Box:
        """Return plot style from a visualization config section."""
        style = Box(
            {
                "color": "red",
                "linestyle": "-",
                "linewidth": cfg.linewidth,
                "label": method,
                "alpha": cfg.alpha,
            }
        )
        if "method_styles" not in cfg:
            return style

        for method_key in (method, method.replace("-", "_")):
            if method_key in cfg.method_styles:
                style.update(cfg.method_styles[method_key])
                return style
        return style

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
        return value

    @staticmethod
    def _metrics_for_latex(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            return df

        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(
            [(column, "") for column in df.columns]
        )
        return df


def combineDfs(df_list: list[pd.DataFrame], methods_list: list[str]) -> str:
    df_all = pd.concat(df_list, ignore_index=True)

    num_cols = [c for c in df_all.columns if c != ("Method", "")]
    df_all[num_cols] = df_all[num_cols].apply(pd.to_numeric, errors="coerce")

    df_mean = df_all.groupby(("Method", "")).mean()
    df_std = df_all.groupby(("Method", "")).std()

    df_ms = pd.concat([df_mean, df_std], keys=["mean", "std"]).swaplevel(0, 1)
    df_ms = df_ms.reindex(pd.MultiIndex.from_product([df_mean.index, ["mean", "std"]]))
    df_ms.index.names = ["Method", "Stat"]

    df_ms = df_ms.reindex(methods_list, level="Method")

    mean_rows = df_ms.index.get_level_values("Stat") == "mean"
    std_rows = df_ms.index.get_level_values("Stat") == "std"

    df_ms = df_ms.reset_index()
    df_ms.columns = pd.MultiIndex.from_tuples(
        [c if isinstance(c, tuple) else (c, "") for c in df_ms.columns]
    )
    df_ms = df_ms.drop(columns=[("Stat", "")])
    df_ms.loc[std_rows, ("Method", "")] = ""

    df_latex = df_ms.copy()
    num_cols = [c for c in df_ms.columns if c != ("Method", "")]

    for col in num_cols:
        df_latex[col] = df_ms[col].map(lambda x: f"{x:.4f}")
        df_latex.loc[std_rows, col] = df_ms.loc[std_rows, col].map(
            lambda x: f"({x:.4f})"
        )
        if "Num (%)" in col[1]:
            df_latex.loc[mean_rows, col] = df_ms.loc[mean_rows, col].map(
                lambda x: f"{x * 100:.2f}%"
            )
            df_latex.loc[std_rows, col] = df_ms.loc[std_rows, col].map(
                lambda x: f"({x * 100:.2f}%)"
            )
        if "Time" in col[0]:
            df_latex.loc[mean_rows, col] = df_ms.loc[mean_rows, col].map(
                lambda x: f"{x * 1000:.2f}"
            )
            df_latex.loc[std_rows, col] = df_ms.loc[std_rows, col].map(
                lambda x: f"({x * 1000:.2f})"
            )

    df_latex = df_latex.rename(
        columns={
            "Train Time (s)": "$T_{train}$ (ms)",
            "Test Time (s)": "$T_{test}$ (ms)",
        },
        level=0,
    )

    try:
        df_latex = df_latex.replace("nan", "-")
        df_latex = df_latex.replace("(nan)", "(-)")
        df_latex.loc[
            df_latex[("Method", "")] == "Optimizer (IPOPT)",
            ("Method", ""),
        ] = "IPOPT"
    except Exception:
        pass
    return df_latex


def df2latex(df: pd.DataFrame) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" + "c" * (len(df.columns) - 1),
    )
    latex = add_midrule_multiheaders(latex)
    latex = latex.replace(r"\$", "$").replace(r"\_", "_").replace("%", r"\%")
    print(latex)
    return latex


def add_midrule_multiheaders(latex):
    import re

    lines = latex.splitlines()
    new_lines = []
    inserted = False
    for idx, line in enumerate(lines):
        new_lines.append(line)
        if (not inserted) and ("\\multicolumn" in line) and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if next_line.lstrip().startswith("&"):
                tokens = [t.strip() for t in line.rstrip("\\").split("&")]
                col = 1
                rules = []
                for tok in tokens:
                    m = re.search(r"\\multicolumn\{(\d+)\}", tok)
                    if m:
                        span = int(m.group(1))
                        start, end = col, col + span - 1
                        if span > 1:
                            rules.append(f"\\cmidrule(lr){{{start}-{end}}}")
                        col += span
                    else:
                        col += 1
                if rules:
                    new_lines.append("".join(rules))
                inserted = True
    return "\n".join(new_lines)
