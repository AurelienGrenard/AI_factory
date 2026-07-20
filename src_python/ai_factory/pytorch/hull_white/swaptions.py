"""Batched exact-transition Hull-White swaption Monte Carlo."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import (
    hull_white_bond_coefficients,
    hull_white_integral_variance,
    hull_white_state_integral_covariance,
    hull_white_state_variance,
    nelson_siegel_discount,
)
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_BATCH_ROWS = 128
MAX_PAYMENTS = 20


def _column(values: list[float], target: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _price_batch_impl(
    rows: list[dict[str, Any]],
    models: dict[str, Any],
    curves: dict[str, Any],
    products: dict[str, Any],
    *,
    num_paths: int,
    target: torch.device,
    batch_rows: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    payment_indices = torch.arange(
        1, MAX_PAYMENTS + 1, device=target, dtype=torch.float64
    )[None, :]

    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        batch_models = [models[row["model_id"]] for row in batch]
        batch_curves = [curves[row["curve_id"]] for row in batch]
        batch_products = [products[row["product_id"]] for row in batch]

        mean_reversion = _column(
            [model["mean_reversion"] for model in batch_models], target
        )
        volatility = _column(
            [model["volatility"] for model in batch_models], target
        )
        beta0 = _column([curve["beta0"] for curve in batch_curves], target)
        beta1 = _column([curve["beta1"] for curve in batch_curves], target)
        beta2 = _column([curve["beta2"] for curve in batch_curves], target)
        tau = _column([curve["tau"] for curve in batch_curves], target)
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

        maturities = expiry + accrual * payment_indices
        active_payments = payment_indices <= payment_count
        bond_a, bond_b = hull_white_bond_coefficients(
            expiry,
            maturities,
            mean_reversion,
            volatility,
            beta0,
            beta1,
            beta2,
            tau,
        )

        normals = torch.randn(
            (len(batch), num_paths, 2), device=target, dtype=torch.float64
        )
        state_variance = hull_white_state_variance(
            mean_reversion, volatility, expiry
        )
        integral_variance = hull_white_integral_variance(
            mean_reversion, volatility, expiry
        )
        covariance = hull_white_state_integral_covariance(
            mean_reversion, volatility, expiry
        )
        state_scale = torch.sqrt(state_variance)
        integral_loading = covariance / state_scale
        state = state_scale * normals[:, :, 0]
        state_integral = (
            integral_loading * normals[:, :, 0]
            + torch.sqrt(
                torch.clamp(
                    integral_variance - integral_loading.square(), min=0.0
                )
            )
            * normals[:, :, 1]
        )
        discount = torch.exp(
            torch.log(nelson_siegel_discount(expiry, beta0, beta1, beta2, tau))
            - 0.5 * integral_variance
            - state_integral
        )

        annuity = torch.zeros_like(state)
        end_bond = torch.zeros_like(state)
        for payment in range(max_payments):
            bond = bond_a[:, payment : payment + 1] * torch.exp(
                -bond_b[:, payment : payment + 1] * state
            )
            active = active_payments[:, payment : payment + 1]
            annuity += torch.where(active, accrual * bond, 0.0)
            end_bond = torch.where(payment_count == payment + 1, bond, end_bond)

        payoff = discount * notional * torch.clamp(
            direction * (1.0 - end_bond - strike * annuity), min=0.0
        )
        outputs.extend(price_summary(payoff))

    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    curve_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    device: str,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(
            target,
            shape=(min(batch_rows, len(rows)), num_paths, 2),
            dtype=torch.float64,
        )
        _price_batch_impl(
            rows[:batch_rows],
            model_by_id,
            curve_by_id,
            product_by_id,
            num_paths=num_paths,
            target=target,
            batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows,
        model_by_id,
        curve_by_id,
        product_by_id,
        num_paths=num_paths,
        target=target,
        batch_rows=batch_rows,
    )
