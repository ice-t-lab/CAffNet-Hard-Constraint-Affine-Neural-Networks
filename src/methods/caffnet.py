"""Constraint-affine neural network methods."""

from __future__ import annotations

from typing import Literal

import torch
from box import Box

from src.base.constraint import BaseConstraint
from src.base.net import BaseNet
from src.utils.CAffine import caffine_project


Architecture = Literal["ff", "tf"]


class CAffNet(BaseNet):
    """CAffNet with feedforward or transformer branches."""

    def __init__(
        self,
        cfg: Box,
        constraint: BaseConstraint,
        architecture: Architecture | None = None,
    ) -> None:
        architecture = architecture or cfg.net.default
        name = "CAffNet-TF" if architecture == "tf" else "CAffNet-FF"
        super().__init__(
            cfg=cfg,
            name=name,
            architecture=architecture,
        )
        self.constraint = constraint
        if self.architecture == "tf":
            self.input_network = self.build_transformer_input()
            self.f_network = self.build_transformer_head()
            self.w_network = self.build_transformer_head()
        else:
            self.f_network = self.build_network()
            self.w_network = self.build_network()

    def apply_projection(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Project output to satisfy ``A(x)y <= b(x)``."""
        A, b = self.constraint.caffnet_coefficients(x)
        return caffine_project(y=y, w=w, A=A, b=b)

    def forward(
        self,
        x: torch.Tensor,
        f_nom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[N, y_dim, 1]`` constrained network output."""
        if self.architecture == "tf":
            h = self.input_network(x.squeeze(-1)).unsqueeze(1)
            y = self.f_network(h).unsqueeze(-1)
            w = self.w_network(h).unsqueeze(-1)
        else:
            y = self.forward_branch(self.f_network, x)
            w = self.forward_branch(self.w_network, x)

        if f_nom is not None:
            y = y + f_nom

        return self.apply_projection(x, y, w)


class CAffNetFF(CAffNet):
    """Feedforward CAffNet."""

    def __init__(
        self,
        cfg: Box,
        constraint: BaseConstraint,
    ) -> None:
        super().__init__(
            cfg=cfg,
            constraint=constraint,
            architecture="ff",
        )


class CAffNetTF(CAffNet):
    """Transformer CAffNet."""

    def __init__(
        self,
        cfg: Box,
        constraint: BaseConstraint,
    ) -> None:
        super().__init__(
            cfg=cfg,
            constraint=constraint,
            architecture="tf",
        )
