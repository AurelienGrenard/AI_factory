"""up and in calls under heston."""

from __future__ import annotations

from typing import Any

import torch

from ai_factory.pytorch.common.barrier_calls import price_barrier_call_batch
from ai_factory.pytorch.heston.pathwise import simulate_barrier_state_batch

DEFAULT_BATCH_ROWS = 16


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
    return price_barrier_call_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_barrier_state_batch,
        up=True,
        knock_in=True,
        dtype=dtype,
        batch_rows=batch_rows,
    )
