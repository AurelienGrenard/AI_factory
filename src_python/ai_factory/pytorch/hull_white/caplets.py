"""Batched exact-transition Hull-White caplet Monte Carlo."""

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


def _column(values: list[float], target: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _price_batch_impl(rows, models, curves, products, *, num_paths, target, batch_rows):
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        ms = [models[row["model_id"]] for row in batch]
        cs = [curves[row["curve_id"]] for row in batch]
        ps = [products[row["product_id"]] for row in batch]
        a = _column([model["mean_reversion"] for model in ms], target)
        sigma = _column([model["volatility"] for model in ms], target)
        beta0 = _column([curve["beta0"] for curve in cs], target)
        beta1 = _column([curve["beta1"] for curve in cs], target)
        beta2 = _column([curve["beta2"] for curve in cs], target)
        tau = _column([curve["tau"] for curve in cs], target)
        fixing = _column([product["fixing_time"] for product in ps], target)
        accrual = _column([product["accrual_period"] for product in ps], target)
        strike = _column([product["strike"] for product in ps], target)
        notional = _column([product["notional"] for product in ps], target)

        bond_a, bond_b = hull_white_bond_coefficients(
            fixing, fixing + accrual, a, sigma, beta0, beta1, beta2, tau
        )
        normals = torch.randn(
            (len(batch), num_paths, 2), device=target, dtype=torch.float64
        )
        state_variance = hull_white_state_variance(a, sigma, fixing)
        integral_variance = hull_white_integral_variance(a, sigma, fixing)
        covariance = hull_white_state_integral_covariance(a, sigma, fixing)
        state_scale = torch.sqrt(state_variance)
        integral_loading = covariance / state_scale
        state = state_scale * normals[:, :, 0]
        state_integral = integral_loading * normals[:, :, 0] + torch.sqrt(
            torch.clamp(integral_variance - integral_loading.square(), min=0.0)
        ) * normals[:, :, 1]
        discount = torch.exp(
            torch.log(nelson_siegel_discount(fixing, beta0, beta1, beta2, tau))
            - 0.5 * integral_variance
            - state_integral
        )
        bond = bond_a * torch.exp(-bond_b * state)
        payoff = discount * notional * torch.clamp(
            1.0 - (1.0 + accrual * strike) * bond, min=0.0
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
):
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(min(batch_rows, len(rows)), num_paths, 2), dtype=torch.float64)
        _price_batch_impl(
            rows[:batch_rows], model_by_id, curve_by_id, product_by_id,
            num_paths=num_paths, target=target, batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows, model_by_id, curve_by_id, product_by_id,
        num_paths=num_paths, target=target, batch_rows=batch_rows,
    )
