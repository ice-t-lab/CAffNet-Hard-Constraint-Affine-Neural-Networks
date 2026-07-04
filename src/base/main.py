"""Base experiment entry point."""

from __future__ import annotations

import logging
import os
import platform
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

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
        self.setup_logging()
        self.log_hardware_info()
        self.logger.info("Using device: %s", self.device)
        self.setup()
        run_methods = methods or self.methods
        completed_methods = [self.run_method(method) for method in run_methods]
        self.save_results(completed_methods)
        return self.result

    def setup(self) -> None:
        """Build shared scenario objects."""
        self.seed_everything()
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

        self.seed_everything()
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
        metrics["Train Time (s)"] = train_time / self.cfg.simulation.training.n_epochs

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
        return simulation.evaluate()

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
        model_path = self.model_path(method)
        torch.save(simulation.net.state_dict(), model_path)
        self.logger.info(
            "Saved model weights at epoch %d to %s",
            self.cfg.simulation.training.n_epochs,
            model_path,
        )

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
    def num_params(net: nn.Module) -> str:
        return f"{sum(p.numel() for p in net.parameters()):,}"

    @staticmethod
    def load_cfg(
        cfg_path: str | Path,
        dir_name: str,
        seed: int | None = None,
    ) -> Box:
        """Load a scenario config and fill the standard result directory."""
        cfg = Config().load(cfg_path)
        if seed is not None:
            cfg.simulation.seed = seed
        cfg.result_dir = str(
            Path("results")
            / str(cfg.scenario.id)
            / dir_name
            / f"seed_{cfg.simulation.seed}"
        )
        return cfg

    def seed_everything(self) -> None:
        """Seed Python, NumPy, and torch."""
        import numpy as np

        seed = self.cfg.simulation.seed
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

    def setup_logging(self) -> None:
        """Configure rebuttal-style console and file logging."""
        result_dir = Path(self.cfg.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        file_handler = logging.FileHandler(result_dir / "log.txt")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    def log_hardware_info(self) -> None:
        """Log runtime hardware in the rebuttal style."""
        self.logger.info("Host: %s", platform.node())
        self.logger.info("OS: %s", platform.platform())
        self.logger.info("CPU cores: %s", os.cpu_count())
        self.logger.info(
            "CUDA available: %s (devices=%d)",
            torch.cuda.is_available(),
            torch.cuda.device_count(),
        )
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            self.logger.info(
                "  GPU %d: %s, capability=%s, vram=%.1f GB",
                index,
                props.name,
                (props.major, props.minor),
                props.total_memory / 1024**3,
            )
