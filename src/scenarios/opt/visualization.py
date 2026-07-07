"""OPT result table utilities."""

from __future__ import annotations

import pandas as pd
import torch
from box import Box

from src.base.visualization import BaseVisualization

from .system import System


class Visualization(BaseVisualization):
    """OPT has no default figure; it reports rebuttal-style metric tables."""

    def __init__(self, cfg: Box, system: System) -> None:
        super().__init__(cfg, system)

    def plot_problem(self, show_plot: bool = True):
        raise NotImplementedError("OPT does not define a problem plot.")

    def plot_result(self, *args, **kwargs):
        raise NotImplementedError("OPT does not define a result plot.")

    def show_table(
        self,
        method: str,
        n_eval: int,
        obj_val: torch.Tensor,
        ineq_err: torch.Tensor,
        eq_err: torch.Tensor,
        train_time_store: list[float] | None = None,
        test_time: float | None = None,
    ) -> pd.DataFrame:
        avg_train_time = (
            sum(train_time_store) / len(train_time_store)
            if train_time_store
            else "-"
        )
        n_ineq = ineq_err.shape[1]
        n_eq = eq_err.shape[1]
        table = pd.DataFrame(
            [
                {
                    "Method": method,
                    "Obj. value": (obj_val / n_eval).item(),
                    "Max ineq viol.": (torch.max(ineq_err, dim=1).values.sum() / n_eval).item(),
                    "Mean ineq viol.": (torch.mean(ineq_err, dim=1).sum() / n_eval).item(),
                    "Num ineq viol. (%)": ((ineq_err > 1e-6).sum() / n_eval / n_ineq).item(),
                    "Max eq viol.": (torch.max(eq_err, dim=1).values.sum() / n_eval).item(),
                    "Mean eq viol.": (torch.mean(eq_err, dim=1).sum() / n_eval).item(),
                    "Num eq viol. (%)": ((eq_err > 1e-6).sum() / n_eval / n_eq).item(),
                    "Train Time (s)": avg_train_time,
                    "Test Time (s)": test_time,
                }
            ]
        )
        print(table.to_string(index=False))
        return table
