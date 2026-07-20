"""Black-Scholes American put pricing with Longstaff-Schwartz in PyTorch."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.american_options import laguerre_basis
from ai_factory.pytorch.common.pathwise_products import parameter_tensor
from ai_factory.pytorch.black_scholes.pathwise import generate_paths

DEFAULT_BATCH_ROWS = 8


def simulate_spot_paths_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.device(device)
    row_count = len(rows)
    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=target, dtype=dtype
    )
    rate = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "risk_free_rate",
        source="model",
        device=target,
        dtype=dtype,
    )
    dividend_yield = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "dividend_yield",
        source="model",
        device=target,
        dtype=dtype,
    )
    volatility = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "volatility",
        source="model",
        device=target,
        dtype=dtype,
    )
    maturity = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "maturity",
        source="product",
        device=target,
        dtype=dtype,
    )
    strike = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "strike",
        source="product",
        device=target,
        dtype=dtype,
    )

    normals = torch.randn((row_count, num_paths, num_steps), device=target, dtype=dtype)
    dt = maturity / float(num_steps)
    increments = (
        (rate - dividend_yield - 0.5 * volatility * volatility).view(-1, 1, 1)
        * dt.view(-1, 1, 1)
        + volatility.view(-1, 1, 1) * torch.sqrt(dt).view(-1, 1, 1) * normals
    )
    paths = torch.empty((row_count, num_paths, num_steps + 1), device=target, dtype=dtype)
    paths[:, :, 0] = spot.view(-1, 1)
    paths[:, :, 1:] = spot.view(-1, 1, 1) * torch.exp(torch.cumsum(increments, dim=2))
    return paths, maturity.view(-1, 1), rate.view(-1, 1), strike


def price_from_paths_batch(
    paths: torch.Tensor,
    *,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    rates: torch.Tensor,
) -> list[dict[str, float]]:
    row_count, num_paths, date_count = paths.shape
    num_steps = date_count - 1
    dt = maturities / float(num_steps)
    payoff = torch.clamp(strikes.view(-1, 1, 1) - paths, min=0.0)
    cashflows = payoff[:, :, -1].clone()
    exercise_steps = torch.full(
        (row_count, num_paths),
        float(num_steps),
        device=paths.device,
        dtype=paths.dtype,
    )
    identity = torch.eye(4, device=paths.device, dtype=paths.dtype).unsqueeze(0)
    basis_count = torch.tensor(4, device=paths.device)

    for step in range(num_steps - 1, 0, -1):
        immediate = payoff[:, :, step]
        itm = immediate > 0.0
        design = laguerre_basis(paths[:, :, step] / strikes.view(-1, 1), dim=2)
        target = cashflows * torch.exp(-rates * dt * (exercise_steps - float(step)))
        weights = itm.to(paths.dtype)
        normal = torch.einsum("rpi,rpj,rp->rij", design, design, weights)
        rhs = torch.einsum("rpi,rp,rp->ri", design, target, weights)
        valid = itm.sum(dim=1) > basis_count
        coefficients = torch.linalg.solve(
            normal + identity * torch.finfo(paths.dtype).eps,
            rhs.unsqueeze(2),
        ).squeeze(2)
        continuation = torch.einsum("rpi,ri->rp", design, coefficients)
        exercise = itm & valid.view(-1, 1) & (immediate > continuation)
        cashflows = torch.where(exercise, immediate, cashflows)
        exercise_steps = torch.where(
            exercise,
            torch.full_like(exercise_steps, float(step)),
            exercise_steps,
        )

    discounted = cashflows * torch.exp(-rates * dt * exercise_steps)
    prices = discounted.mean(dim=1)
    stderrs = discounted.std(dim=1, unbiased=True) / math.sqrt(num_paths)
    return [
        {"price": float(price.cpu()), "standard_error": float(stderr.cpu())}
        for price, stderr in zip(prices, stderrs, strict=True)
    ]


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
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = perf_counter()
    simulation_seconds = 0.0
    lsm_seconds = 0.0
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        sim_started = perf_counter()
        paths, maturities, rates, strikes = simulate_spot_paths_batch(
            chunk,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=target,
            dtype=dtype,
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        simulation_seconds += perf_counter() - sim_started
        lsm_started = perf_counter()
        outputs.extend(
            price_from_paths_batch(
                paths,
                strikes=strikes,
                maturities=maturities,
                rates=rates,
            )
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        lsm_seconds += perf_counter() - lsm_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "lsm_seconds": lsm_seconds,
    }


def price_variable_grid_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    step_counts: list[int],
    device: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = 32,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Price heterogeneous grids in padded batches while preserving each row dt."""

    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        chunk_counts = torch.tensor(
            step_counts[start : start + batch_rows],
            device=target_device,
            dtype=torch.long,
        )
        max_steps = int(chunk_counts.max().item())
        spot = parameter_tensor(
            chunk, model_by_id, product_by_id, "spot",
            source="model", device=target_device, dtype=dtype,
        )
        rate = parameter_tensor(
            chunk, model_by_id, product_by_id, "risk_free_rate",
            source="model", device=target_device, dtype=dtype,
        )
        dividend = parameter_tensor(
            chunk, model_by_id, product_by_id, "dividend_yield",
            source="model", device=target_device, dtype=dtype,
        )
        volatility = parameter_tensor(
            chunk, model_by_id, product_by_id, "volatility",
            source="model", device=target_device, dtype=dtype,
        )
        maturity = parameter_tensor(
            chunk, model_by_id, product_by_id, "maturity",
            source="product", device=target_device, dtype=dtype,
        )
        strike = parameter_tensor(
            chunk, model_by_id, product_by_id, "strike",
            source="product", device=target_device, dtype=dtype,
        )
        spot = spot.squeeze(1)
        rate = rate.squeeze(1)
        dividend = dividend.squeeze(1)
        volatility = volatility.squeeze(1)
        maturity = maturity.squeeze(1)
        strike = strike.squeeze(1)
        dt = maturity / chunk_counts.to(dtype)
        normals = torch.randn(
            (len(chunk), num_paths, max_steps),
            device=target_device,
            dtype=dtype,
        )
        increments = (
            (rate - dividend - 0.5 * volatility.square()).view(-1, 1, 1)
            * dt.view(-1, 1, 1)
            + volatility.view(-1, 1, 1)
            * torch.sqrt(dt).view(-1, 1, 1)
            * normals
        )
        paths = torch.empty(
            (len(chunk), num_paths, max_steps + 1),
            device=target_device,
            dtype=dtype,
        )
        paths[:, :, 0] = spot.view(-1, 1)
        paths[:, :, 1:] = spot.view(-1, 1, 1) * torch.exp(
            torch.cumsum(increments, dim=2)
        )

        terminal_index = chunk_counts.view(-1, 1, 1).expand(-1, num_paths, 1)
        terminal_spot = paths.gather(2, terminal_index).squeeze(2)
        cashflows = torch.clamp(strike.view(-1, 1) - terminal_spot, min=0.0)
        exercise_steps = chunk_counts.view(-1, 1).expand(-1, num_paths).to(dtype).clone()
        identity = torch.eye(4, device=target_device, dtype=dtype).unsqueeze(0)
        basis_count = torch.tensor(4, device=target_device)
        for step in range(max_steps - 1, 0, -1):
            active = step < chunk_counts.view(-1, 1)
            immediate = torch.clamp(
                strike.view(-1, 1) - paths[:, :, step], min=0.0
            )
            itm = active & (immediate > 0.0)
            design = laguerre_basis(
                paths[:, :, step] / strike.view(-1, 1), dim=2
            )
            regression_target = cashflows * torch.exp(
                -rate.view(-1, 1) * dt.view(-1, 1)
                * (exercise_steps - float(step))
            )
            weights = itm.to(dtype)
            normal = torch.einsum("rpi,rpj,rp->rij", design, design, weights)
            rhs = torch.einsum(
                "rpi,rp,rp->ri", design, regression_target, weights
            )
            valid = itm.sum(dim=1) > basis_count
            coefficients = torch.linalg.solve(
                normal + identity * torch.finfo(dtype).eps,
                rhs.unsqueeze(2),
            ).squeeze(2)
            continuation = torch.einsum("rpi,ri->rp", design, coefficients)
            exercise = itm & valid.view(-1, 1) & (immediate > continuation)
            cashflows = torch.where(exercise, immediate, cashflows)
            exercise_steps = torch.where(
                exercise, torch.full_like(exercise_steps, float(step)), exercise_steps
            )

        discounted = cashflows * torch.exp(
            -rate.view(-1, 1) * dt.view(-1, 1) * exercise_steps
        )
        prices = discounted.mean(dim=1).cpu().tolist()
        stderrs = (
            discounted.std(dim=1, unbiased=True) / math.sqrt(num_paths)
        ).cpu().tolist()
        outputs.extend(
            {"price": float(price), "standard_error": float(stderr)}
            for price, stderr in zip(prices, stderrs, strict=True)
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    return outputs, {"wall_seconds": perf_counter() - started}


def price(
    model_parameters: dict[str, float],
    product_parameters: dict[str, Any],
    simulation: dict[str, Any],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    paths = generate_paths(
        model_parameters,
        product_parameters,
        simulation,
        device=device,
        dtype=dtype,
    )
    target = torch.device(device)
    strikes = torch.tensor([float(product_parameters["strike"])], device=target, dtype=dtype)
    maturities = torch.tensor([[float(product_parameters["maturity"])]], device=target, dtype=dtype)
    rates = torch.tensor([[float(model_parameters["risk_free_rate"])]], device=target, dtype=dtype)
    return price_from_paths_batch(
        paths.unsqueeze(0),
        strikes=strikes,
        maturities=maturities,
        rates=rates,
    )[0]
