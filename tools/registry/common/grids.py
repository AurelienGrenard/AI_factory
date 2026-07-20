"""Helpers to build product parameter grids."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def exponential_strike_grid(
    *,
    spot: float,
    maturities: Sequence[float],
    strikes_per_maturity: int,
    log_moneyness_width: float,
) -> list[dict[str, float]]:
    """Build a maturity-dependent exponential strike grid.

    Strikes are generated as K = S0 * exp(x), where x is a log-moneyness grid
    over [-width * sqrt(T), width * sqrt(T)]. Long maturities therefore get a
    wider strike range than short maturities.
    """

    if strikes_per_maturity < 1:
        raise ValueError("strikes_per_maturity must be at least 1.")
    if spot <= 0.0:
        raise ValueError("spot must be positive.")
    if log_moneyness_width < 0.0:
        raise ValueError("log_moneyness_width must be non-negative.")

    products: list[dict[str, float]] = []
    for maturity in maturities:
        if maturity <= 0.0:
            raise ValueError("maturities must be positive.")

        radius = log_moneyness_width * math.sqrt(maturity)
        if strikes_per_maturity == 1:
            offsets = [0.0]
        else:
            offsets = [
                -radius + 2.0 * radius * index / (strikes_per_maturity - 1)
                for index in range(strikes_per_maturity)
            ]

        for offset in offsets:
            products.append(
                {
                    "strike": spot * math.exp(offset),
                    "maturity": maturity,
                }
            )

    return products


def add_numeric_ids(
    parameters: Sequence[dict[str, Any]],
    *,
    width: int = 6,
) -> list[dict[str, Any]]:
    """Attach zero-padded local ids to parameter dictionaries."""

    return [
        {
            "id": str(index).zfill(width),
            "parameters": dict(parameter),
        }
        for index, parameter in enumerate(parameters, start=1)
    ]
