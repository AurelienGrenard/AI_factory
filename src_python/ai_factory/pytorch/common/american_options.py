"""PyTorch building blocks shared by American-option pricers."""

from __future__ import annotations

import math

import torch


def laguerre_basis(values: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    x = values
    return torch.stack(
        [
            torch.ones_like(x),
            1.0 - x,
            1.0 - 2.0 * x + 0.5 * x * x,
            1.0 - 3.0 * x + 1.5 * x * x - x * x * x / 6.0,
        ],
        dim=dim,
    )


def price_puts_from_paths_batch(
    paths: torch.Tensor,
    *,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    rates: torch.Tensor,
    regression_features: torch.Tensor | None = None,
    ridge_relative: float = 1.0e-10,
) -> list[dict[str, float]]:
    """Apply LSM to spot paths and optional model-state features.

    The common basis starts with ``[1, L1(S/K), L2(S/K)]``. Additional
    features must have shape ``(rows, paths, dates, features)``.
    """

    row_count, num_paths, date_count = paths.shape
    num_steps = date_count - 1
    dt = maturities / float(num_steps)
    payoff = torch.clamp(strikes.unsqueeze(1) - paths, min=0.0)
    cashflows = payoff[:, :, -1].clone()
    exercise_steps = torch.full(
        (row_count, num_paths), float(num_steps),
        device=paths.device, dtype=paths.dtype,
    )
    feature_count = 0 if regression_features is None else regression_features.shape[-1]
    basis_count = 3 + feature_count
    identity = torch.eye(
        basis_count, device=paths.device, dtype=paths.dtype
    ).unsqueeze(0)

    for step in range(num_steps - 1, 0, -1):
        immediate = payoff[:, :, step]
        itm = immediate > 0.0
        spot_basis = laguerre_basis(paths[:, :, step] / strikes, dim=2)[:, :, :3]
        design = (
            spot_basis
            if regression_features is None
            else torch.cat(
                (spot_basis, regression_features[:, :, step, :]), dim=2
            )
        )
        target = cashflows * torch.exp(
            -rates * dt * (exercise_steps - float(step))
        )
        weights = itm.to(paths.dtype)
        normal = torch.einsum("rpi,rpj,rp->rij", design, design, weights)
        rhs = torch.einsum("rpi,rp,rp->ri", design, target, weights)
        valid = itm.sum(dim=1) > basis_count
        ridge = torch.clamp(
            ridge_relative
            * normal.diagonal(dim1=1, dim2=2).sum(dim=1)
            / float(basis_count),
            min=torch.finfo(paths.dtype).eps,
        )
        coefficients = torch.linalg.solve(
            normal + identity * ridge[:, None, None],
            rhs.unsqueeze(2),
        ).squeeze(2)
        continuation = torch.einsum("rpi,ri->rp", design, coefficients)
        exercise = itm & valid.view(-1, 1) & (immediate > continuation)
        cashflows = torch.where(exercise, immediate, cashflows)
        exercise_steps = torch.where(
            exercise, torch.full_like(exercise_steps, float(step)), exercise_steps
        )

    discounted = cashflows * torch.exp(-rates * dt * exercise_steps)
    prices = discounted.mean(dim=1)
    stderrs = discounted.std(dim=1, unbiased=True) / math.sqrt(num_paths)
    return [
        {"price": float(price.cpu()), "standard_error": float(stderr.cpu())}
        for price, stderr in zip(prices, stderrs, strict=True)
    ]
