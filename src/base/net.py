"""Base network utilities shared by method implementations."""

from __future__ import annotations

from typing import Literal

import torch
from box import Box
from torch import nn


Architecture = Literal["ff", "tf"]


class TransformerBlock(nn.Module):
    """Transformer encoder block used by transformer-based methods."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or d_model

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer block on token tensor ``[N, T, d_model]``."""
        attention, _ = self.attention(x, x, x)
        x = self.norm_attention(x + self.dropout(attention))
        feedforward = self.feedforward(x)
        return self.norm_feedforward(x + self.dropout(feedforward))

    @staticmethod
    def activation(name: str) -> nn.Module:
        """Return an activation module."""
        if name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {name}")


class TransformerRegressor(nn.Module):
    """Vector-to-vector transformer branch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.transformer = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model,
            activation=activation,
            dropout=dropout,
        )
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[N, output_dim]`` from vector input ``[N, input_dim]``."""
        h = self.input(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        return self.output(h)


class BaseNet(nn.Module):
    """Common network-building helpers for all methods."""

    def __init__(
        self,
        cfg: Box,
        name: str,
        architecture: Architecture | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.architecture = architecture or cfg.net.default
        self.x_dim = int(cfg.system.x_dim)
        self.y_dim = int(cfg.system.y_dim)

    def build_network(
        self,
        output_dim: int | None = None,
        architecture: Architecture | None = None,
    ) -> nn.Module:
        """Build a feedforward or transformer branch."""
        output_dim = output_dim or self.y_dim
        architecture = architecture or self.architecture

        if architecture == "ff":
            return self.build_feedforward(output_dim)
        if architecture == "tf":
            return self.build_transformer(output_dim)
        raise ValueError(f"Unsupported architecture: {architecture}")

    def build_feedforward(self, output_dim: int | None = None) -> nn.Sequential:
        """Build an MLP from ``cfg.net.ff``."""
        output_dim = output_dim or self.y_dim
        hidden_dims = list(self.cfg.net.ff.hidden_dims)
        activation = self.cfg.net.ff.activation
        dropout = float(getattr(self.cfg.net.ff, "dropout", 0.0))

        layers: list[nn.Module] = []
        in_dim = self.x_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(in_dim, int(hidden_dim))
            self.init_linear(linear)
            layers.append(linear)
            layers.append(self.activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = int(hidden_dim)

        output = nn.Linear(in_dim, output_dim)
        self.init_linear(output)
        layers.append(output)
        return nn.Sequential(*layers)

    def build_transformer(self, output_dim: int | None = None) -> TransformerRegressor:
        """Build a transformer branch from ``cfg.net.tf``."""
        output_dim = output_dim or self.y_dim
        net = TransformerRegressor(
            input_dim=self.x_dim,
            output_dim=output_dim,
            d_model=int(self.cfg.net.tf.d_model),
            num_heads=int(self.cfg.net.tf.num_heads),
            activation=self.cfg.net.tf.activation,
            dropout=float(getattr(self.cfg.net.tf, "dropout", 0.0)),
        )
        self.init_module(net)
        return net

    def forward_branch(self, branch: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a branch and return ``[N, output_dim, 1]``."""
        return branch(x.squeeze(-1)).unsqueeze(-1)

    @staticmethod
    def activation(name: str) -> nn.Module:
        """Return an activation module."""
        if name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unsupported activation: {name}")

    def init_module(self, module: nn.Module) -> None:
        """Initialize all linear and attention layers in a module."""
        module.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        """Initialize layers with Xavier initialization."""
        if isinstance(module, nn.Linear):
            self.init_linear(module)
        elif isinstance(module, nn.MultiheadAttention):
            self.init_attention(module)

    def init_linear(self, layer: nn.Linear) -> None:
        """Initialize a linear layer."""
        nn.init.xavier_normal_(layer.weight)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def init_attention(self, layer: nn.MultiheadAttention) -> None:
        """Initialize a multihead attention layer."""
        nn.init.xavier_normal_(layer.in_proj_weight)
        nn.init.xavier_normal_(layer.out_proj.weight)

        if layer.in_proj_bias is not None:
            nn.init.zeros_(layer.in_proj_bias)
        if layer.out_proj.bias is not None:
            nn.init.zeros_(layer.out_proj.bias)
