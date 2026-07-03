"""Shared single-network training and evaluation loop."""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn.functional as F
from box import Box
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .constraint import BaseConstraint
from .system import BaseSystem


Batch = dict[str, torch.Tensor]
LossTerms = dict[str, torch.Tensor]
Metrics = dict[str, float]


class BaseSimulation:
    """Common training/evaluation code for one network."""

    def __init__(
        self,
        cfg: Box,
        system: BaseSystem,
        net: nn.Module,
        constraint: BaseConstraint | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.system = system
        self.constraint = constraint
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.cfg.simulation.training.lr)
        self.loss_history: list[Metrics] = []
        self.logger = logging.getLogger(__name__)

    def train(self, data: Batch | None = None) -> None:
        """Train the network."""
        train_data = self.get_data("train") if data is None else data
        n_epochs = self.cfg.simulation.training.n_epochs

        for epoch in range(1, n_epochs + 1):
            self.net.train()
            totals: Metrics = {}
            n_batches = 0
            batch_size = self.cfg.simulation.training.batch_size

            for batch in self.iter_batches(train_data, batch_size, shuffle=True):
                self.optimizer.zero_grad()
                terms = self.simulate(batch)
                terms["total_loss"].backward()
                self.optimizer.step()

                n_batches += 1
                self.accumulate(totals, terms)

            terms = self.average(totals, n_batches)
            self.loss_history.append({"epoch": float(epoch), **terms})

            if self.should_log(epoch, n_epochs):
                self.logger.info(
                    "%s epoch %s/%s: %s",
                    self.net.name,
                    epoch,
                    n_epochs,
                    self.format_terms(terms),
                )

    def evaluate(self, data: Batch | None = None) -> Metrics:
        """Evaluate the network."""
        eval_data = self.get_data("eval") if data is None else data
        self.net.eval()

        totals: Metrics = {}
        n_batches = 0
        batch_size = self.cfg.simulation.training.batch_size

        with torch.no_grad():
            for batch in self.iter_batches(eval_data, batch_size, shuffle=False):
                terms = self.simulate(batch)

                n_batches += 1
                self.accumulate(totals, terms)

        return self.average(totals, n_batches)

    def get_data(self, split: Literal["train", "eval"]) -> Batch:
        """Return unified x/y data for train or eval."""
        if split == "train":
            return {"x": self.system.x_train, "y": self.system.y_train}
        return {"x": self.system.x_eval, "y": self.system.y_eval}

    def simulate(self, batch: Batch) -> LossTerms:
        """Run one batch simulation and return loss terms.

        Subclasses can override this for rollout-based systems, such as CBF
        robotic simulations that need multiple system steps per training batch.
        """
        y_pred = self.net(batch["x"])
        return self.loss(batch, y_pred)

    def loss(
        self,
        batch: Batch,
        y_pred: torch.Tensor,
    ) -> LossTerms:
        """Main loss plus weighted constraint violation."""
        main = self.main_loss(batch, y_pred)
        violation = self.constraint_violation_loss(batch, y_pred)
        soft_weight = self.cfg.simulation.training.soft_weight
        total = main + soft_weight * violation
        return {
            "total_loss": total,
            "main_loss": main,
            "constraint_violation_loss": violation,
        }

    def main_loss(self, batch: Batch, y_pred: torch.Tensor) -> torch.Tensor:
        """Default supervised MSE loss."""
        return F.mse_loss(y_pred, batch["y"])

    def constraint_violation_loss(
        self,
        batch: Batch,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared positive affine-constraint residual."""
        residual = self.system.constraint_residual(
            batch["x"],
            y_pred,
            self.constraint,
            self.net.name,
        )
        if residual is None:
            return y_pred.new_zeros(())
        return residual.square().mean()

    def iter_batches(
        self,
        data: Batch,
        batch_size: int,
        shuffle: bool,
    ):
        """Yield mini-batches using the same DataLoader path as rebuttal."""
        n_samples = len(data["x"])
        if batch_size <= 0:
            batch_size = n_samples

        keys = list(data)
        dataset = TensorDataset(*(data[key] for key in keys))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for tensors in dataloader:
            yield {
                key: tensor.to(self.device)
                for key, tensor in zip(keys, tensors, strict=True)
            }

    def should_log(self, epoch: int, n_epochs: int) -> bool:
        """Whether this epoch should be logged."""
        freq = self.cfg.simulation.logging.log_freq
        return epoch == 1 or epoch == n_epochs or epoch % freq == 0

    @staticmethod
    def accumulate(totals: Metrics, terms: LossTerms) -> None:
        """Accumulate scalar tensor terms."""
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())

    @staticmethod
    def average(totals: Metrics, n_batches: int) -> Metrics:
        """Average batch totals."""
        if n_batches == 0:
            return {}
        return {key: value / n_batches for key, value in totals.items()}

    @staticmethod
    def format_terms(terms: Metrics) -> str:
        """Format terms for logging."""
        return ", ".join(f"{key}={value:.6g}" for key, value in terms.items())
