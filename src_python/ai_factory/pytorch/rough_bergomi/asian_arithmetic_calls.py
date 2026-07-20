"""Rough Bergomi arithmetic Asian call pricing in PyTorch."""

from __future__ import annotations

from typing import Any

import torch

from ai_factory.pytorch.common.pathwise_products import (
    call_delta_crn_batch,
    call_price_batch,
)
from ai_factory.pytorch.rough_bergomi.pathwise import simulate_average_spot_batch

DEFAULT_BATCH_ROWS = 8


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
    return call_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_average_spot_batch,
        strike_key="strike",
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
    return call_delta_crn_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_average_spot_batch,
        strike_key="strike",
        relative_bump=relative_bump,
        dtype=dtype,
        batch_rows=batch_rows,
    )
