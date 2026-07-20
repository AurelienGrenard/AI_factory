"""Tensorized Longstaff-Schwartz backward induction for Bermudan claims."""

from __future__ import annotations

import torch


def _basis(state: torch.Tensor) -> torch.Tensor:
    square = state.square()
    return torch.stack(
        (
            torch.ones_like(state),
            1.0 - state,
            1.0 - 2.0 * state + 0.5 * square,
            1.0 - 3.0 * state + 1.5 * square - square * state / 6.0,
        ),
        dim=-1,
    )


def present_value_cashflows(
    immediate: torch.Tensor,
    basis_state: torch.Tensor,
    discounts: torch.Tensor,
    exercise_count: torch.Tensor,
) -> torch.Tensor:
    """Return one time-zero exercised cashflow per row and path."""
    row_count, path_count, max_exercises = immediate.shape
    row_index = torch.arange(row_count, device=immediate.device)
    last = exercise_count - 1
    gather_index = last[:, None, None].expand(-1, path_count, 1)
    cashflows = torch.gather(discounts * immediate, 2, gather_index).squeeze(2)

    for exercise in range(max_exercises - 2, -1, -1):
        active_row = exercise < last
        exercise_value = immediate[:, :, exercise]
        itm = (exercise_value > 0.0) & active_row[:, None]
        basis = _basis(basis_state[:, :, exercise])
        weights = itm.to(immediate.dtype)
        target = cashflows / discounts[:, :, exercise]
        normal = torch.einsum("bpi,bpj,bp->bij", basis, basis, weights)
        rhs = torch.einsum("bpi,bp,bp->bi", basis, target, weights)
        scale = torch.diagonal(normal, dim1=-2, dim2=-1).amax(dim=-1)
        ridge = torch.finfo(immediate.dtype).eps * torch.clamp(scale, min=1.0)
        normal = normal + ridge[:, None, None] * torch.eye(
            4, device=immediate.device, dtype=immediate.dtype
        )
        coefficients = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
        continuation = torch.einsum("bpi,bi->bp", basis, coefficients)
        valid = itm.sum(dim=1) > 4
        exercise_now = itm & valid[:, None] & (exercise_value > continuation)
        replacement = discounts[:, :, exercise] * exercise_value
        cashflows = torch.where(exercise_now, replacement, cashflows)
    return cashflows
