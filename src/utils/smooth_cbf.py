"""Smooth CBF set operations."""

from __future__ import annotations

import torch


def smooth_intersection(
    h_i: torch.Tensor,
    Lfh_i: torch.Tensor,
    Lgh_i: torch.Tensor,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = -torch.logsumexp(-kappa * h_i, dim=1, keepdim=True) / kappa
    weights = torch.softmax(-kappa * h_i, dim=1)
    Lfh = torch.sum(weights * Lfh_i, dim=1, keepdim=True)
    Lgh = torch.sum(weights * Lgh_i, dim=1, keepdim=True)
    return h, Lfh, Lgh


def smooth_union(
    h_i: torch.Tensor,
    Lfh_i: torch.Tensor,
    Lgh_i: torch.Tensor,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_constraints = h_i.shape[1]
    logits = kappa * h_i
    h = torch.logsumexp(logits, dim=1, keepdim=True) / kappa
    h = h - torch.log(
        torch.tensor(n_constraints, dtype=h_i.dtype, device=h_i.device)
    ) / kappa
    weights = torch.softmax(logits, dim=1)
    Lfh = torch.sum(weights * Lfh_i, dim=1, keepdim=True)
    Lgh = torch.sum(weights * Lgh_i, dim=1, keepdim=True)
    return h, Lfh, Lgh
