"""Common Monte Carlo reductions for PyTorch pricing engines."""

from __future__ import annotations

import math

import torch


def price_summary(discounted: torch.Tensor) -> list[dict[str, float]]:
    """Return price and standard error for one row per matrix row."""

    prices = discounted.mean(dim=1)
    stderrs = discounted.std(dim=1, unbiased=True) / math.sqrt(discounted.shape[1])
    values = torch.stack((prices, stderrs), dim=1).cpu().tolist()
    return [
        {"price": price, "standard_error": stderr}
        for price, stderr in values
    ]


def price_delta_summary(
    discounted: torch.Tensor,
    delta_paths: torch.Tensor,
) -> list[dict[str, float]]:
    """Return price, delta, and corresponding Monte Carlo standard errors."""

    prices = discounted.mean(dim=1)
    price_stderrs = discounted.std(dim=1, unbiased=True) / math.sqrt(discounted.shape[1])
    deltas = delta_paths.mean(dim=1)
    delta_stderrs = delta_paths.std(dim=1, unbiased=True) / math.sqrt(delta_paths.shape[1])
    values = torch.stack(
        (prices, price_stderrs, deltas, delta_stderrs), dim=1
    ).cpu().tolist()
    return [
        {
            "price": price,
            "standard_error": price_stderr,
            "delta": delta,
            "delta_standard_error": delta_stderr,
        }
        for price, price_stderr, delta, delta_stderr in values
    ]
