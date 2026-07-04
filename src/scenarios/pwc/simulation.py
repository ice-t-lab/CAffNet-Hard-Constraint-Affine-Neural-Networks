"""PWC training loop."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.base.simulation import BaseSimulation, Batch, LossTerms, Metrics


class Simulation(BaseSimulation):
    """Use rebuttal-style summed training losses for PWC."""

    def before_train(self) -> None:
        """Reset RNG before DataLoader iteration to match rebuttal."""
        seed = self.cfg.simulation.seed
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
        """Run one PWC batch with rebuttal-style summed losses."""
        y_pred = self.predict(batch)
        main_loss = self.system.get_main_loss(batch["y"], y_pred)
        constraint_violation_loss = self.system.get_constraint_violation_loss(
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
        """Return rebuttal-style PWC evaluation metrics."""
        n_eval = batch["x"].shape[0]
        mse = self.system.get_main_loss(batch["y"], y_pred) / n_eval
        ineq_err = self.system.get_ineq_err(batch["x"], y_pred)
        return {
            "MSE": self.to_scalar(mse),
            "Max ineq viol.": self.to_scalar(
                torch.max(ineq_err, dim=1).values.sum() / n_eval
            ),
            "Mean ineq viol.": self.to_scalar(
                torch.mean(ineq_err, dim=1).sum() / n_eval
            ),
            "Test Time (s)": test_time,
        }
