"""Pathwise adapters for Rough Bergomi PyTorch products."""

from __future__ import annotations

from typing import Any

import torch

from ai_factory.pytorch.common.pathwise_products import (
    chunk_inputs,
    discount_tensor,
    parameter_tensor,
)
from ai_factory.pytorch.rough_bergomi.common import (
    generate_average_spots_batch,
    generate_max_spots_batch,
    generate_observation_spots_batch,
    generate_realized_volatilities_batch,
    generate_terminal_extrema_batch,
)


def simulate_observation_spots_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    return generate_observation_spots_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )


def simulate_barrier_state_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    state = generate_terminal_extrema_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )
    return state[:, :, 0], state[:, :, 2 if up else 1]


def simulate_max_spot_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    max_spots = generate_max_spots_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )
    return max_spots, _discount(rows, model_by_id, product_by_id, device, dtype), _spot(
        rows, model_by_id, product_by_id, device, dtype
    )


def simulate_terminal_spot_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    terminal = generate_terminal_extrema_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )[:, :, 0]
    return terminal, _discount(rows, model_by_id, product_by_id, device, dtype), _spot(
        rows, model_by_id, product_by_id, device, dtype
    )


def simulate_average_spot_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    average_spots = generate_average_spots_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )
    return average_spots, _discount(rows, model_by_id, product_by_id, device, dtype), _spot(
        rows, model_by_id, product_by_id, device, dtype
    )


def simulate_realized_volatility_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    models, products, simulations = chunk_inputs(
        rows, model_by_id, product_by_id, num_paths, num_steps
    )
    realized_volatilities = generate_realized_volatilities_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )
    return (
        realized_volatilities,
        _discount(rows, model_by_id, product_by_id, device, dtype),
        _spot(rows, model_by_id, product_by_id, device, dtype),
    )


def _discount(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return discount_tensor(
        rows,
        model_by_id,
        product_by_id,
        device=device,
        dtype=dtype,
    )


def _spot(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "spot",
        source="model",
        device=device,
        dtype=dtype,
    )
