"""Shared batched pricing for discretely monitored barrier calls."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.monte_carlo import price_summary
from ai_factory.pytorch.common.pathwise_products import parameter_tensor

BarrierSimulation = Callable[
    [list[dict[str, Any]], dict[str, Any], dict[str, Any], int, int, torch.device, torch.dtype, bool],
    tuple[torch.Tensor, torch.Tensor],
]


def price_barrier_call_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: BarrierSimulation,
    up: bool,
    knock_in: bool,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    _sync(target)
    started = perf_counter()
    simulation_seconds = 0.0
    payoff_seconds = 0.0
    outputs: list[dict[str, float]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_rows):
            chunk = rows[start : start + batch_rows]
            sim_started = perf_counter()
            terminal, extremum = simulate_batch(
                chunk, model_by_id, product_by_id, num_paths, num_steps, target, dtype, up
            )
            _sync(target)
            simulation_seconds += perf_counter() - sim_started

            payoff_started = perf_counter()
            strike = parameter_tensor(
                chunk, model_by_id, product_by_id, "strike",
                source="product", device=target, dtype=dtype,
            )
            barrier = parameter_tensor(
                chunk, model_by_id, product_by_id, "barrier",
                source="product", device=target, dtype=dtype,
            )
            rate = parameter_tensor(
                chunk, model_by_id, product_by_id, "risk_free_rate",
                source="model", device=target, dtype=dtype,
            )
            maturity = parameter_tensor(
                chunk, model_by_id, product_by_id, "maturity",
                source="product", device=target, dtype=dtype,
            )
            hit = extremum >= barrier if up else extremum <= barrier
            active = hit if knock_in else ~hit
            discounted = (
                torch.exp(-rate * maturity)
                * torch.clamp(terminal - strike, min=0.0)
                * active.to(dtype)
            )
            outputs.extend(price_summary(discounted))
            _sync(target)
            payoff_seconds += perf_counter() - payoff_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "payoff_seconds": payoff_seconds,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
