"""Shared PyTorch pricing wrappers for terminal call payoffs."""

from __future__ import annotations

from typing import Any

import torch

from ai_factory.pytorch.common.pathwise_products import (
    call_price_batch,
    digital_call_price_batch,
)


def price_call_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return call_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_batch,
        strike_key="strike",
        dtype=torch.float64,
        batch_rows=batch_rows,
    )


def price_digital_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return digital_call_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_batch,
        strike_key="strike",
        dtype=torch.float64,
        batch_rows=batch_rows,
    )
