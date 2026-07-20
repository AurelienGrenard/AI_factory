"""Batched CIR QE swaption Monte Carlo."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import cir_bond_coefficients
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_CPU_BATCH_ROWS = 32
DEFAULT_CUDA_BATCH_ROWS = 64
MAX_PAYMENTS = 20
QE_PSI_CUTOFF = 1.5


def _column(values: list[float], target: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _step_count(product: dict[str, Any], target_dt: float) -> int:
    return max(1, round(float(product["expiry"]) / target_dt))


def _price_group(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    products: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int | None,
    target_dt: float,
    target: torch.device,
    batch_rows: int,
) -> list[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    payment_indices = torch.arange(
        1, MAX_PAYMENTS + 1, device=target, dtype=torch.float64
    )[None, :]

    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        batch_models = [models[row["model_id"]] for row in batch]
        batch_products = [products[row["product_id"]] for row in batch]

        kappa = _column([model["kappa"] for model in batch_models], target)
        theta = _column([model["theta"] for model in batch_models], target)
        volatility = _column(
            [model["volatility"] for model in batch_models], target
        )
        initial_rate = _column(
            [model["initial_rate"] for model in batch_models], target
        )
        expiry = _column([product["expiry"] for product in batch_products], target)
        accrual = _column(
            [product["accrual_period"] for product in batch_products], target
        )
        strike = _column(
            [product["fixed_rate"] for product in batch_products], target
        )
        notional = _column(
            [product["notional"] for product in batch_products], target
        )
        direction = _column(
            [product["direction"] for product in batch_products], target
        )
        payment_count = torch.tensor(
            [product["payment_count"] for product in batch_products],
            device=target,
            dtype=torch.int64,
        )[:, None]
        max_payments = max(
            int(product["payment_count"]) for product in batch_products
        )

        horizons = accrual * payment_indices
        active_payments = payment_indices <= payment_count
        bond_a, bond_b = cir_bond_coefficients(
            kappa, theta, volatility, horizons
        )

        rate = initial_rate.expand(-1, num_paths).clone()
        integral = torch.zeros_like(rate)
        if num_steps is None:
            host_step_counts = [
                _step_count(product, target_dt) for product in batch_products
            ]
            max_steps = max(host_step_counts)
            step_counts = torch.tensor(
                host_step_counts, device=target, dtype=torch.int64
            )[:, None]
        else:
            max_steps = num_steps
            step_counts = torch.full(
                (len(batch), 1), num_steps, device=target, dtype=torch.int64
            )
        dt = expiry / step_counts
        decay = torch.exp(-kappa * dt)
        one_minus_decay = 1.0 - decay
        volatility_squared = volatility.square()

        for step in range(max_steps):
            previous_rate = rate
            mean = theta + (rate - theta) * decay
            variance = (
                rate
                * volatility_squared
                * decay
                * one_minus_decay
                / kappa
                + theta
                * volatility_squared
                * one_minus_decay.square()
                / (2.0 * kappa)
            )
            psi = variance / mean.square()
            normal = torch.randn_like(rate)
            uniform = torch.rand_like(rate)

            inverse_psi = 1.0 / torch.clamp(psi, min=1.0e-14)
            b_squared = (
                2.0 * inverse_psi
                - 1.0
                + torch.sqrt(2.0 * inverse_psi)
                * torch.sqrt(torch.clamp(2.0 * inverse_psi - 1.0, min=0.0))
            )
            quadratic = (
                mean
                / (1.0 + b_squared)
                * (torch.sqrt(torch.clamp(b_squared, min=0.0)) + normal).square()
            )
            probability = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - probability) / mean
            exponential = torch.where(
                uniform <= probability,
                torch.zeros_like(rate),
                torch.log((1.0 - probability) / (1.0 - uniform)) / beta,
            )
            proposed_rate = torch.where(
                psi <= QE_PSI_CUTOFF, quadratic, exponential
            )
            active_step = step < step_counts
            rate = torch.where(active_step, proposed_rate, previous_rate)
            integral += torch.where(
                active_step,
                0.5 * (previous_rate + proposed_rate) * dt,
                0.0,
            )

        annuity = torch.zeros_like(rate)
        end_bond = torch.zeros_like(rate)
        for payment in range(max_payments):
            bond = bond_a[:, payment : payment + 1] * torch.exp(
                -bond_b[:, payment : payment + 1] * rate
            )
            active = active_payments[:, payment : payment + 1]
            annuity += torch.where(active, accrual * bond, 0.0)
            end_bond = torch.where(payment_count == payment + 1, bond, end_bond)

        payoff = torch.exp(-integral) * notional * torch.clamp(
            direction * (1.0 - end_bond - strike * annuity), min=0.0
        )
        outputs.extend(price_summary(payoff))

    return outputs


def _price_batch_impl(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    products: dict[str, Any],
    *,
    num_paths: int,
    target: torch.device,
    target_dt: float,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    started = perf_counter()
    if target.type == "cuda":
        outputs = _price_group(
            rows,
            models,
            products,
            num_paths=num_paths,
            num_steps=None,
            target_dt=target_dt,
            target=target,
            batch_rows=batch_rows,
        )
        synchronize(target)
        return outputs, {"wall_seconds": perf_counter() - started}

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_step_count(products[row["product_id"]], target_dt)].append(
            (index, row)
        )

    ordered: list[dict[str, float] | None] = [None] * len(rows)
    for num_steps, indexed_rows in grouped.items():
        group_outputs = _price_group(
            [row for _, row in indexed_rows],
            models,
            products,
            num_paths=num_paths,
            num_steps=num_steps,
            target_dt=target_dt,
            target=target,
            batch_rows=batch_rows,
        )
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            ordered[index] = output

    synchronize(target)
    return [output for output in ordered if output is not None], {
        "wall_seconds": perf_counter() - started
    }


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    device: str,
    target_dt: float = 1.0 / 52.0,
    batch_rows: int | None = None,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = resolve_device(device)
    if batch_rows is None:
        batch_rows = (
            DEFAULT_CUDA_BATCH_ROWS
            if target.type == "cuda"
            else DEFAULT_CPU_BATCH_ROWS
        )
    if target.type == "cuda" and rows:
        warmup(
            target,
            shape=(min(batch_rows, len(rows)), num_paths),
            dtype=torch.float64,
        )
        _price_batch_impl(
            rows[:batch_rows],
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            target=target,
            target_dt=target_dt,
            batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        target=target,
        target_dt=target_dt,
        batch_rows=batch_rows,
    )
