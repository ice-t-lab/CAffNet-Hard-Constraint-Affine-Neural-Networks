"""Projection utilities for constraint-affine networks."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def project_to_active_constraints(
    y: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Project ``y`` onto active affine constraints.

    Args:
        y: [N, y_dim, 1] network output.
        w: [N, y_dim, 1] learned null-space bias.
        A: [N, n_combinations, n_active, y_dim] active constraints.
        b: [N, n_combinations, n_active, 1] active bounds.

    Returns:
        [N, n_combinations, y_dim, 1] projected candidates.
    """
    batch_size, n_combinations, _, y_dim = A.shape
    y = y.unsqueeze(1).expand(batch_size, n_combinations, y_dim, 1)
    w = w.unsqueeze(1).expand(batch_size, n_combinations, y_dim, 1)

    A_flat = A.reshape(batch_size * n_combinations, -1, y_dim)
    A_pinv = torch.linalg.pinv(A_flat).reshape(batch_size, n_combinations, y_dim, -1)
    eye = torch.eye(y_dim, dtype=y.dtype, device=y.device).expand(
        batch_size,
        n_combinations,
        y_dim,
        y_dim,
    )

    return y - A_pinv @ (A @ y - b) + (eye - A_pinv @ A) @ w


def constraint_combinations(
    A: torch.Tensor,
    b: torch.Tensor,
    n_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all combinations of ``n_active`` rows from ``A`` and ``b``.

    Args:
        A: [N, n_constraints, y_dim].
        b: [N, n_constraints, 1].
        n_active: number of active constraints in each combination.
    """
    batch_size, n_constraints, y_dim = A.shape
    if n_active > n_constraints:
        raise ValueError(
            f"n_active={n_active} must be <= n_constraints={n_constraints}."
        )

    combinations = list(itertools.combinations(range(n_constraints), n_active))
    A_selected = A.new_zeros(batch_size, len(combinations), n_active, y_dim)
    b_selected = b.new_zeros(batch_size, len(combinations), n_active, 1)

    for i, rows in enumerate(combinations):
        idx = torch.tensor(rows, device=A.device)
        A_selected[:, i] = A[:, idx]
        b_selected[:, i] = b[:, idx]

    return A_selected, b_selected


def select_feasible_projection(
    y: torch.Tensor,
    candidates: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-10,
    penalty: float = 1e10,
) -> torch.Tensor:
    """Choose the closest feasible projection candidate.

    Args:
        y: [N, y_dim, 1] original output.
        candidates: [N, n_candidates, y_dim, 1] projection candidates.
        A: [N, n_constraints, y_dim].
        b: [N, n_constraints, 1].
    """
    batch_size, n_candidates, y_dim, _ = candidates.shape
    n_constraints = A.shape[1]

    A = A.unsqueeze(1).expand(batch_size, n_candidates, n_constraints, y_dim)
    b = b.unsqueeze(1).expand(batch_size, n_candidates, n_constraints, 1)
    y = y.unsqueeze(1).expand(batch_size, n_candidates, y_dim, 1)

    invalid = (A @ candidates - b >= tol).any(dim=2)
    distance = torch.norm(y - candidates, dim=2) + penalty * invalid
    idx = torch.argmin(distance, dim=1)
    return candidates.gather(
        dim=1,
        index=idx.view(batch_size, 1, 1, 1).expand(batch_size, 1, y_dim, 1),
    ).squeeze(1)


def caffine_project(
    y: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-10,
    penalty: float = 1e10,
) -> torch.Tensor:
    """Project ``y`` to satisfy CAffNet constraints ``A(x)y <= b(x)``.

    This mirrors the rebuttal CAffNet projection: include the original output
    as a candidate, enumerate active sets up to ``min(y_dim, n_constraints)``,
    then choose the closest feasible projection.
    """
    _, n_constraints, y_dim = A.shape
    candidates = [y.unsqueeze(1)]

    for n_active in range(1, min(y_dim, n_constraints) + 1):
        A_active, b_active = constraint_combinations(A, b, n_active)
        candidates.append(project_to_active_constraints(y, w, A_active, b_active))

    return select_feasible_projection(
        y=y,
        candidates=torch.cat(candidates, dim=1),
        A=A,
        b=b,
        tol=tol,
        penalty=penalty,
    )


def hardnet_project(
    y: torch.Tensor,
    A: torch.Tensor,
    bl: torch.Tensor,
    bu: torch.Tensor,
) -> torch.Tensor:
    """Project ``y`` to satisfy HardNet constraints ``bl <= A(x)y <= bu``."""
    Ay = A @ y
    correction = F.relu(bl - Ay) - F.relu(Ay - bu)
    return y + torch.linalg.lstsq(A, correction).solution
