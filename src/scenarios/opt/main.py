"""Run the OPT scenario."""

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
    """OPT experiment runner."""

    default_methods = ["NN", "HardNet", "CAffNet-FF", "CAffNet-TF", "Optimizer (IPOPT)"]

    def build_system(self) -> System:
        return System(self.cfg)

    def build_constraint(self) -> Constraint:
        return Constraint(self.cfg, self.system)

    def build_visualization(self) -> Visualization:
        return Visualization(self.cfg, self.system)

    def build_simulation(self, net: torch.nn.Module) -> BaseSimulation:
        return Simulation(
            cfg=self.cfg,
            system=self.system,
            net=net,
            constraint=self.constraint,
            device=self.device,
        )

    def run_method(self, method: str) -> str:
        if method in ("IPOPT", "Optimizer (IPOPT)"):
            return self.run_ipopt()
        return super().run_method(method)

    def run_ipopt(self) -> str:
        method = "Optimizer (IPOPT)"
        self.logger.info("Method: %s", method)
        x_eval = self.system.x_eval.to(self.device)
        y_eval, test_time = self.system.solve_ipopt(x_eval)
        n_eval = x_eval.shape[0]
        obj_val = self.system.get_main_loss(None, y_eval)
        ineq_err = self.system.get_ineq_err(x_eval, y_eval)
        eq_err = self.system.get_eq_err(x_eval, y_eval)
        metrics = {
            "Obj. value": (obj_val / n_eval).item(),
            "Max ineq viol.": (torch.max(ineq_err, dim=1).values.sum() / n_eval).item(),
            "Mean ineq viol.": (torch.mean(ineq_err, dim=1).sum() / n_eval).item(),
            "Num ineq viol. (%)": ((ineq_err > 1e-6).sum() / n_eval / ineq_err.shape[1]).item(),
            "Max eq viol.": (torch.max(eq_err, dim=1).values.sum() / n_eval).item(),
            "Mean eq viol.": (torch.mean(eq_err, dim=1).sum() / n_eval).item(),
            "Num eq viol. (%)": ((eq_err > 1e-6).sum() / n_eval / eq_err.shape[1]).item(),
            "Test Time (s)": test_time,
        }
        self.result.save(
            method=method,
            metrics=metrics if self.cfg.simulation.save.metrics else None,
            result_data={"x": x_eval.detach().cpu(), "y": y_eval.detach().cpu()},
            loss_history=None,
        )
        return method


def main(
    dir_name: str,
    method: str,
    seed: int | None = None,
) -> None:
    cfg = Main.load_cfg(Path(__file__).with_name("cfg.yaml"), dir_name, seed)
    methods = None if method == "all" else [method]
    Main(cfg, methods=methods).run()


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
