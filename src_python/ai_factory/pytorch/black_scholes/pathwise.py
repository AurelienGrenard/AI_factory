"""Black-Scholes path simulation utilities in PyTorch."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import (
    resolve_device,
    seeded_generator,
    synchronize as _synchronize_if_needed,
)
from ai_factory.pytorch.common.pathwise_products import parameter_tensor


def simulate_terminal_spot_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the exact Black-Scholes terminal transition in one vectorized launch."""

    del num_steps
    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=device, dtype=dtype
    )
    rate = parameter_tensor(
        rows, model_by_id, product_by_id, "risk_free_rate", source="model", device=device, dtype=dtype
    )
    dividend = parameter_tensor(
        rows, model_by_id, product_by_id, "dividend_yield", source="model", device=device, dtype=dtype
    )
    volatility = parameter_tensor(
        rows, model_by_id, product_by_id, "volatility", source="model", device=device, dtype=dtype
    )
    maturity = parameter_tensor(
        rows, model_by_id, product_by_id, "maturity", source="product", device=device, dtype=dtype
    )
    shocks = torch.randn((len(rows), num_paths), device=device, dtype=dtype)
    terminal = spot * torch.exp(
        (rate - dividend - 0.5 * volatility.square()) * maturity
        + volatility * torch.sqrt(maturity) * shocks
    )
    return terminal, torch.exp(-rate * maturity), spot


def generate_paths(
    model_parameters: Mapping[str, float],
    product_parameters: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    output_device = resolve_device(device)
    seed = int(simulation["seed"])
    num_paths = int(simulation["num_paths"])
    num_steps = int(simulation["num_steps"])
    maturity = float(product_parameters["maturity"])

    spot = float(model_parameters["spot"])
    rate = float(model_parameters["risk_free_rate"])
    dividend_yield = float(model_parameters.get("dividend_yield", 0.0))
    volatility = float(model_parameters["volatility"])

    dt = maturity / num_steps
    generator = seeded_generator(seed, output_device)
    shocks = torch.randn(
        (num_paths, num_steps),
        generator=generator,
        device=output_device,
        dtype=dtype,
    )
    increments = (
        (rate - dividend_yield - 0.5 * volatility * volatility) * dt
        + volatility * dt**0.5 * shocks
    )
    log_paths = torch.empty(
        (num_paths, num_steps + 1),
        device=output_device,
        dtype=dtype,
    )
    log_paths[:, 0] = torch.log(torch.as_tensor(spot, device=output_device, dtype=dtype))
    log_paths[:, 1:] = log_paths[:, :1] + torch.cumsum(increments, dim=1)
    return torch.exp(log_paths)


def simulate_statistic_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    statistic: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if statistic not in {"max_spot", "average_spot", "realized_volatility"}:
        raise ValueError(f"Unsupported Black-Scholes statistic: {statistic}")

    row_count = len(rows)
    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=device, dtype=dtype
    )
    rate = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "risk_free_rate",
        source="model",
        device=device,
        dtype=dtype,
    )
    dividend_yield = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "dividend_yield",
        source="model",
        device=device,
        dtype=dtype,
    )
    volatility = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "volatility",
        source="model",
        device=device,
        dtype=dtype,
    )
    maturity = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "maturity",
        source="product",
        device=device,
        dtype=dtype,
    )
    seed = torch.tensor(
        [int(row["seed"]) for row in rows],
        device=device,
        dtype=torch.int64,
    )

    start = perf_counter()
    # PyTorch generators are scalar objects, so rows are simulated in row batches,
    # while each row still uses a fully vectorized path matrix on the target device.
    values = []
    for index in range(row_count):
        generator = seeded_generator(int(seed[index].item()), device)
        shocks = torch.randn(
            (num_paths, num_steps),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        dt = maturity[index] / float(num_steps)
        increments = (
            (rate[index] - dividend_yield[index] - 0.5 * volatility[index] * volatility[index]) * dt
            + volatility[index] * torch.sqrt(dt) * shocks
        )
        paths = torch.empty((num_paths, num_steps + 1), device=device, dtype=dtype)
        paths[:, 0] = spot[index]
        paths[:, 1:] = spot[index] * torch.exp(torch.cumsum(increments, dim=1))
        if statistic == "max_spot":
            values.append(paths.max(dim=1).values)
        elif statistic == "average_spot":
            values.append(paths[:, 1:].mean(dim=1))
        else:
            log_returns = torch.diff(torch.log(paths), dim=1)
            values.append(torch.sqrt(torch.sum(log_returns * log_returns, dim=1) / maturity[index]))

    statistics = torch.stack(values, dim=0)
    _synchronize_if_needed(device)
    _ = perf_counter() - start
    discount = torch.exp(-rate * maturity)
    return statistics, discount, spot


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
    """Simulate only the quarterly spot observations required by autocalls."""

    target = torch.device(device)
    row_count = len(rows)
    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=target, dtype=dtype
    )
    rate = parameter_tensor(
        rows, model_by_id, product_by_id, "risk_free_rate", source="model", device=target, dtype=dtype
    )
    dividend_yield = parameter_tensor(
        rows, model_by_id, product_by_id, "dividend_yield", source="model", device=target, dtype=dtype
    )
    volatility = parameter_tensor(
        rows, model_by_id, product_by_id, "volatility", source="model", device=target, dtype=dtype
    )
    maturity = parameter_tensor(
        rows, model_by_id, product_by_id, "maturity", source="product", device=target, dtype=dtype
    )
    observation_counts = {
        int(product_by_id[row["product_id"]]["observation_count"]) for row in rows
    }
    if len(observation_counts) != 1:
        raise ValueError("A PyTorch autocall batch must share one observation count.")
    observation_count = observation_counts.pop()
    if num_steps % observation_count != 0:
        raise ValueError("Observation count must divide num_steps.")
    stride = num_steps // observation_count
    shocks = torch.randn((row_count, num_paths, num_steps), device=target, dtype=dtype)
    dt = maturity / float(num_steps)
    increments = (
        (rate - dividend_yield - 0.5 * volatility.square()).view(-1, 1, 1)
        * dt.view(-1, 1, 1)
        + volatility.view(-1, 1, 1)
        * torch.sqrt(dt).view(-1, 1, 1)
        * shocks
    )
    log_spots = torch.log(spot).view(-1, 1, 1) + torch.cumsum(increments, dim=2)
    return torch.exp(log_spots[:, :, stride - 1 :: stride])


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
    """Return terminal spots and monitored extrema without materializing S0."""

    row_count = len(rows)
    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=device, dtype=dtype
    )
    rate = parameter_tensor(
        rows, model_by_id, product_by_id, "risk_free_rate", source="model", device=device, dtype=dtype
    )
    dividend = parameter_tensor(
        rows, model_by_id, product_by_id, "dividend_yield", source="model", device=device, dtype=dtype
    )
    volatility = parameter_tensor(
        rows, model_by_id, product_by_id, "volatility", source="model", device=device, dtype=dtype
    )
    maturity = parameter_tensor(
        rows, model_by_id, product_by_id, "maturity", source="product", device=device, dtype=dtype
    )
    shocks = torch.randn((row_count, num_paths, num_steps), device=device, dtype=dtype)
    dt = maturity / float(num_steps)
    increments = (
        (rate - dividend - 0.5 * volatility.square())[:, None] * dt[:, None]
        + volatility[:, None] * torch.sqrt(dt)[:, None] * shocks
    )
    log_paths = torch.log(spot)[:, None] + torch.cumsum(increments, dim=2)
    paths = torch.exp(log_paths)
    terminal = paths[:, :, -1]
    extremum = paths.amax(dim=2) if up else paths.amin(dim=2)
    return terminal, extremum
