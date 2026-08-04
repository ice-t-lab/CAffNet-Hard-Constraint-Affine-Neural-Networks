"""OPT training and evaluation loop."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.base.simulation import BaseSimulation, Batch, LossTerms, Metrics


class Simulation(BaseSimulation):
    """Use objective value plus soft affine-constraint violation for OPT."""

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

    def simulate(self, batch: Batch) -> LossTerms:
        y_pred = self.predict(batch)
        main_loss = self.system.get_main_loss(batch.get("y"), y_pred)
        constraint_violation_loss = self.constraint.violation_loss(
            batch["x"],
            y_pred,
        )
        total_loss = (
            main_loss
            + self.cfg.simulation.training.soft_weight * constraint_violation_loss
        )
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "constraint_violation_loss": constraint_violation_loss,
        }

    def loss(
        self,
        batch: Batch,
        y_pred: torch.Tensor,
    ) -> LossTerms:
        main_loss = self.system.get_main_loss(batch.get("y"), y_pred)
        constraint_violation_loss = self.constraint.violation_loss(
            batch["x"],
            y_pred,
        )
        total_loss = (
            main_loss
            + self.cfg.simulation.training.soft_weight * constraint_violation_loss
        )
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "constraint_violation_loss": constraint_violation_loss,
        }

    def evaluation_metrics(
        self,
        batch: Batch,
        y_pred: torch.Tensor,
        terms: LossTerms,
        test_time: float,
    ) -> Metrics:
        return self.metrics(batch["x"], y_pred, test_time)

    def metrics(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        test_time: float,
    ) -> Metrics:
        n_eval = x.shape[0]
        obj_val = self.system.get_main_loss(None, y)
        ineq_err = self.constraint.ineq_err(x, y)
        eq_err = self.constraint.eq_err(x, y)
        n_ineq = ineq_err.shape[1]
        n_eq = eq_err.shape[1]
        return {
            "Obj. value": self.to_scalar(obj_val / n_eval),
            "Max ineq viol.": self.to_scalar(torch.max(ineq_err, dim=1).values.sum() / n_eval),
            "Mean ineq viol.": self.to_scalar(torch.mean(ineq_err, dim=1).sum() / n_eval),
            "Num ineq viol. (%)": self.to_scalar((ineq_err > 1e-6).sum() / n_eval / n_ineq),
            "Max eq viol.": self.to_scalar(torch.max(eq_err, dim=1).values.sum() / n_eval),
            "Mean eq viol.": self.to_scalar(torch.mean(eq_err, dim=1).sum() / n_eval),
            "Num eq viol. (%)": self.to_scalar((eq_err > 1e-6).sum() / n_eval / n_eq),
            "Test Time (s)": test_time,
        }
