"""Shared PyTorch wrappers for pathwise Monte Carlo products."""

from __future__ import annotations

import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.monte_carlo import price_delta_summary, price_summary

PathwiseSimulation = Callable[
    [
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        int,
        int,
        torch.device,
        torch.dtype,
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def parameter_tensor(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    name: str,
    *,
    source: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for row in rows:
        if source == "model":
            values.append(float(model_by_id[row["model_id"]][name]))
        elif source == "product":
            values.append(float(product_by_id[row["product_id"]][name]))
        else:
            raise ValueError(f"Unsupported batched parameter source: {source}")
    return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)


def chunk_inputs(
    chunk_rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    models = [model_by_id[row["model_id"]] for row in chunk_rows]
    products = [product_by_id[row["product_id"]] for row in chunk_rows]
    simulations = [
        {"seed": row["seed"], "num_paths": num_paths, "num_steps": num_steps}
        for row in chunk_rows
    ]
    return models, products, simulations


def discount_tensor(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        math.exp(
            -float(model_by_id[row["model_id"]]["risk_free_rate"])
            * float(product_by_id[row["product_id"]]["maturity"])
        )
        for row in rows
    ]
    return torch.tensor(values, device=device, dtype=dtype).view(-1, 1)


def call_price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: PathwiseSimulation,
    strike_key: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return _price_batches(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        batch_rows=batch_rows,
        simulate_batch=simulate_batch,
        payoff_kind="call",
        strike_key=strike_key,
    )


def digital_call_price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: PathwiseSimulation,
    strike_key: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    _sync(target)
    started = perf_counter()
    simulation_seconds = 0.0
    payoff_seconds = 0.0
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        sim_started = perf_counter()
        terminal, discount, _ = simulate_batch(
            chunk,
            model_by_id,
            product_by_id,
            num_paths,
            num_steps,
            target,
            dtype,
        )
        _sync(target)
        simulation_seconds += perf_counter() - sim_started
        payoff_started = perf_counter()
        strikes = parameter_tensor(
            chunk,
            model_by_id,
            product_by_id,
            strike_key,
            source="product",
            device=target,
            dtype=dtype,
        )
        outputs.extend(price_summary(discount * (terminal > strikes).to(dtype)))
        _sync(target)
        payoff_seconds += perf_counter() - payoff_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "payoff_seconds": payoff_seconds,
    }


def call_delta_crn_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: PathwiseSimulation,
    strike_key: str,
    relative_bump: float,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    _sync(target)
    started = perf_counter()
    simulation_seconds = 0.0
    payoff_seconds = 0.0
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        sim_started = perf_counter()
        statistic, discount, spot = simulate_batch(
            chunk,
            model_by_id,
            product_by_id,
            num_paths,
            num_steps,
            target,
            dtype,
        )
        _sync(target)
        simulation_seconds += perf_counter() - sim_started
        payoff_started = perf_counter()
        strikes = parameter_tensor(
            chunk,
            model_by_id,
            product_by_id,
            strike_key,
            source="product",
            device=target,
            dtype=dtype,
        )
        payoff = torch.clamp(statistic - strikes, min=0.0)
        up_payoff = torch.clamp(statistic * (1.0 + relative_bump) - strikes, min=0.0)
        down_payoff = torch.clamp(statistic * (1.0 - relative_bump) - strikes, min=0.0)
        delta_paths = (
            discount * (up_payoff - down_payoff) / (2.0 * relative_bump * spot)
        )
        outputs.extend(price_delta_summary(discount * payoff, delta_paths))
        _sync(target)
        payoff_seconds += perf_counter() - payoff_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "payoff_seconds": payoff_seconds,
    }


def linear_price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: PathwiseSimulation,
    strike_key: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    return _price_batches(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        batch_rows=batch_rows,
        simulate_batch=simulate_batch,
        payoff_kind="linear",
        strike_key=strike_key,
    )


def zero_delta_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_batch: PathwiseSimulation,
    strike_key: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    outputs, timing = linear_price_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        simulate_batch=simulate_batch,
        strike_key=strike_key,
        dtype=dtype,
        batch_rows=batch_rows,
    )
    for output in outputs:
        output["delta"] = 0.0
        output["delta_standard_error"] = 0.0
    return outputs, timing


def _price_batches(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    dtype: torch.dtype,
    batch_rows: int,
    simulate_batch: PathwiseSimulation,
    payoff_kind: str,
    strike_key: str,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    _sync(target)
    started = perf_counter()
    simulation_seconds = 0.0
    payoff_seconds = 0.0
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        sim_started = perf_counter()
        statistic, discount, _ = simulate_batch(
            chunk,
            model_by_id,
            product_by_id,
            num_paths,
            num_steps,
            target,
            dtype,
        )
        _sync(target)
        simulation_seconds += perf_counter() - sim_started
        payoff_started = perf_counter()
        strikes = parameter_tensor(
            chunk,
            model_by_id,
            product_by_id,
            strike_key,
            source="product",
            device=target,
            dtype=dtype,
        )
        if payoff_kind == "call":
            discounted = discount * torch.clamp(statistic - strikes, min=0.0)
        elif payoff_kind == "linear":
            discounted = discount * (statistic - strikes)
        else:
            raise ValueError(f"Unsupported payoff kind: {payoff_kind}")
        outputs.extend(price_summary(discounted))
        _sync(target)
        payoff_seconds += perf_counter() - payoff_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "payoff_seconds": payoff_seconds,
    }


def _sync(target: torch.device) -> None:
    if target.type == "cuda":
        torch.cuda.synchronize(target)
