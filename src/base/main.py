"""Base experiment entry point."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from box import Box
from torch import nn

from src.methods.caffnet import CAffNetFF, CAffNetTF
from src.methods.hardnet import HardNet
from src.methods.nn import NN
from src.utils.config import Config

from .constraint import BaseConstraint
from .result import BaseResult, Metrics, ResultData
from .simulation import BaseSimulation
from .system import BaseSystem
from .visualization import BaseVisualization


class BaseMain(ABC):
    """Shared train, evaluate, save, and result workflow."""

    default_methods = ["NN", "HardNet", "CAffNet-FF", "CAffNet-TF"]

    def __init__(
        self,
        cfg: Box,
        methods: list[str] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.methods = methods or self.default_methods
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logging.getLogger(__name__)

        self.system: BaseSystem | None = None
        self.constraint: BaseConstraint | None = None
        self.visualization: BaseVisualization | None = None
        self.result: BaseResult | None = None

    def run(self, methods: list[str] | None = None) -> BaseResult:
        """Run methods and build saved result artifacts."""
        self.setup()
        run_methods = methods or self.methods
        completed_methods = [self.run_method(method) for method in run_methods]
        self.save_results(completed_methods)
        return self.result

    def setup(self) -> None:
        """Build shared scenario objects."""
        self.system = self.build_system()
        self.constraint = self.build_constraint()
        self.visualization = self.build_visualization()
        self.result = self.build_result()

        if self.cfg.simulation.save.config:
            Config().save(self.cfg, self.result.result_dir / "cfg.yaml")
        self.save_common_data()

    def run_method(self, method: str) -> str:
        """Train, evaluate, and save one method."""
        self.logger.info("Method: %s", method)

        net = self.build_net(method)
        method = net.name
        simulation = self.build_simulation(net)

        self.logger.info(net)
        self.logger.info("Total parameters: %s", self.num_params(net))

        train_start = time.perf_counter()
        simulation.train()
        train_time = time.perf_counter() - train_start

        if self.cfg.simulation.save.model_state:
            self.save_model(method, simulation)
        if self.cfg.simulation.save.optimizer_state:
            self.save_optimizer(method, simulation)

        metrics, result_data = self.evaluate(simulation)
        metrics["Train Time (s)"] = train_time

        self.result.save(
            method=method,
            metrics=metrics if self.cfg.simulation.save.metrics else None,
            result_data=result_data,
            loss_history=(
                simulation.loss_history
                if self.cfg.simulation.save.loss_history
                else None
            ),
        )
        return method

    def evaluate(
        self,
        simulation: BaseSimulation,
    ) -> tuple[Metrics, ResultData]:
        """Evaluate once and return saved metrics plus plotting data."""
        data = simulation.get_data("eval")
        batch = {key: value.to(simulation.device) for key, value in data.items()}

        simulation.net.eval()
        with torch.no_grad():
            test_start = time.perf_counter()
            y_pred = simulation.net(batch["x"])
            test_time = time.perf_counter() - test_start
            terms = simulation.loss(batch, y_pred)

        metrics = self.metrics_from_terms(terms)
        metrics["Test Time (s)"] = test_time
        result_data = {
            "x": batch["x"].detach().cpu(),
            "y": y_pred.detach().cpu(),
        }
        return metrics, result_data

    def save_results(self, methods: list[str]) -> None:
        """Build combined result outputs from saved method data."""
        if self.cfg.simulation.save.metrics:
            self.result.table(methods)
            self.result.latex_table(methods)

        if self.cfg.simulation.save.figures and "result" in self.cfg.visualization:
            self.result.plot_results(methods)
        if (
            self.cfg.simulation.save.figures
            and self.cfg.simulation.save.loss_history
            and "loss_history" in self.cfg.visualization
        ):
            self.result.plot_loss(methods)

    def save_common_data(self) -> None:
        """Save data shared by all methods."""
        if self.cfg.simulation.save.training_samples:
            torch.save(
                {"x": self.system.x_train, "y": self.system.y_train},
                self.result.result_dir / "train_data.pt",
            )
        if self.cfg.simulation.save.evaluation_samples:
            torch.save(
                {"x": self.system.x_eval, "y": self.system.y_eval},
                self.result.result_dir / "eval_data.pt",
            )

    def build_net(self, method: str) -> nn.Module:
        """Build a supported network method."""
        if method == "NN":
            return NN(self.cfg)
        if method == "HardNet":
            return HardNet(self.cfg, self.require_constraint(method))
        if method in ("CAffNet-FF", "CAffNet_FF", "CAffNetFF"):
            return CAffNetFF(self.cfg, self.require_constraint(method))
        if method in ("CAffNet-TF", "CAffNet_TF", "CAffNetTF"):
            return CAffNetTF(self.cfg, self.require_constraint(method))
        raise ValueError(f"Unknown method: {method}")

    @abstractmethod
    def build_system(self) -> BaseSystem:
        """Build the scenario system."""
        raise NotImplementedError

    def build_constraint(self) -> BaseConstraint | None:
        """Build scenario constraints when a method needs them."""
        return None

    def build_visualization(self) -> BaseVisualization:
        """Build scenario visualization."""
        return BaseVisualization(self.cfg, self.system)

    def build_simulation(self, net: nn.Module) -> BaseSimulation:
        """Build simulation/training loop for one net."""
        return BaseSimulation(
            cfg=self.cfg,
            system=self.system,
            net=net,
            constraint=self.constraint,
            device=self.device,
        )

    def build_result(self) -> BaseResult:
        """Build saved-result handler."""
        return BaseResult(self.cfg, self.visualization)

    def save_model(self, method: str, simulation: BaseSimulation) -> None:
        """Save network state dict inside the method folder."""
        self.result.method_dir(method).mkdir(parents=True, exist_ok=True)
        torch.save(simulation.net.state_dict(), self.model_path(method))

    def save_optimizer(self, method: str, simulation: BaseSimulation) -> None:
        """Save optimizer state dict inside the method folder."""
        self.result.method_dir(method).mkdir(parents=True, exist_ok=True)
        torch.save(simulation.optimizer.state_dict(), self.optimizer_path(method))

    def model_path(self, method: str) -> Path:
        return self.result.method_dir(method) / "model.pt"

    def optimizer_path(self, method: str) -> Path:
        return self.result.method_dir(method) / "optimizer.pt"

    def require_constraint(self, method: str) -> BaseConstraint:
        if self.constraint is None:
            raise ValueError(f"{method} requires a constraint.")
        return self.constraint

    @staticmethod
    def metrics_from_terms(terms: dict[str, torch.Tensor]) -> Metrics:
        """Convert scalar tensor loss terms to saved metrics."""
        return {
            key: BaseMain.to_scalar(value)
            for key, value in terms.items()
        }

    @staticmethod
    def to_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
        return value

    @staticmethod
    def num_params(net: nn.Module) -> str:
        return f"{sum(p.numel() for p in net.parameters()):,}"
