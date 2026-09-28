"""Automatic differentiation helpers shared by training and evaluation."""

from __future__ import annotations

import torch


def selected_diagonal_hessians(
    first_derivatives: torch.Tensor,
    inputs: torch.Tensor,
    input_indices: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return selected diagonal entries of d(first_derivatives)/d(inputs)."""

    columns: list[torch.Tensor] = []
    for index in input_indices.detach().cpu().tolist():
        component = first_derivatives[:, index]
        if component.requires_grad:
            second_derivatives = torch.autograd.grad(
                component.sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            second_derivatives = None
        columns.append(
            component * 0.0
            if second_derivatives is None
            else second_derivatives[:, index]
        )
    if not columns:
        return first_derivatives.new_empty((first_derivatives.shape[0], 0))
    return torch.stack(columns, dim=1)
