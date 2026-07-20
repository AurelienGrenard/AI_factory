"""Heston volatility swap pricing in PyTorch."""

from __future__ import annotations

from typing import Any

import torch

from ai_factory.pytorch.common.pathwise_products import (
    linear_price_batch,
    zero_delta_batch,
)
from ai_factory.pytorch.heston.pathwise import simulate_statistic_batch

DEFAULT_BATCH_ROWS = 16


def _simulate_realized_volatility(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return simulate_statistic_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        statistic="realized_volatility",
    )


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return linear_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=_simulate_realized_volatility,
        strike_key="volatility_strike",
        dtype=dtype,
        batch_rows=batch_rows,
    )


def price_delta_crn_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    relative_bump: float,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    del relative_bump
    return zero_delta_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=_simulate_realized_volatility,
        strike_key="volatility_strike",
        dtype=dtype,
        batch_rows=batch_rows,
    )
