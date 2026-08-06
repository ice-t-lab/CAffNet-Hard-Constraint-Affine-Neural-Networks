"""Run the PWC scenario."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch

from src.base.main import BaseMain
from src.base.simulation import BaseSimulation

from .constraint import Constraint
from .simulation import Simulation
from .system import System
from .visualization import Visualization


torch.set_default_dtype(torch.float64)


class Main(BaseMain):
    """PWC experiment runner."""

    def build_system(self) -> System:
        return System(self.cfg)

    def build_constraint(self) -> Constraint:
        return Constraint(self.cfg, self.system)

    def build_visualization(self) -> Visualization:
        return Visualization(self.cfg, self.system, self.constraint)

    def build_simulation(self, net: torch.nn.Module) -> BaseSimulation:
        return Simulation(
            cfg=self.cfg,
            system=self.system,
            net=net,
            constraint=self.constraint,
            device=self.device,
        )


def main(
    dir_name: str,
    method: str,
    seed: int | None = None,
) -> None:
    cfg = Main.load_cfg(Path(__file__).with_name("cfg.yaml"), dir_name, seed)
    Main(cfg, methods=[method]).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name",
        type=str,
        default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--method", type=str, default="NN")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args.dir_name, args.method, args.seed)
