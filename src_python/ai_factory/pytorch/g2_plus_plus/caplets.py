"""Batched exact-transition G2++ caplet Monte Carlo."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import (
    g2_bond_coefficients,
    g2_path_discount,
    g2_transition,
)
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_BATCH_ROWS = 128


def _column(values, target):
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _price_batch_impl(rows, models, curves, products, *, num_paths, target, batch_rows):
    started = perf_counter()
    outputs = []
    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        ms = [models[row["model_id"]] for row in batch]
        cs = [curves[row["curve_id"]] for row in batch]
        ps = [products[row["product_id"]] for row in batch]
        a = _column([model["mean_reversion_x"] for model in ms], target)
        sigma = _column([model["volatility_x"] for model in ms], target)
        b = _column([model["mean_reversion_y"] for model in ms], target)
        eta = _column([model["volatility_y"] for model in ms], target)
        rho = _column([model["rho"] for model in ms], target)
        beta0 = _column([curve["beta0"] for curve in cs], target)
        beta1 = _column([curve["beta1"] for curve in cs], target)
        beta2 = _column([curve["beta2"] for curve in cs], target)
        tau = _column([curve["tau"] for curve in cs], target)
        fixing = _column([product["fixing_time"] for product in ps], target)
        accrual = _column([product["accrual_period"] for product in ps], target)
        strike = _column([product["strike"] for product in ps], target)
        notional = _column([product["notional"] for product in ps], target)

        bond_a, bond_x, bond_y = g2_bond_coefficients(
            fixing, fixing + accrual, a, sigma, b, eta, rho,
            beta0, beta1, beta2, tau,
        )
        *_, cholesky = g2_transition(a, sigma, b, eta, rho, fixing)
        normals = torch.randn(
            (len(batch), num_paths, 4), device=target, dtype=torch.float64
        )
        x, y, integral_x, integral_y = torch.matmul(
            normals, cholesky.transpose(-1, -2)
        ).unbind(-1)
        bond = bond_a * torch.exp(-bond_x * x - bond_y * y)
        discount = g2_path_discount(
            integral_x, integral_y, fixing, a, sigma, b, eta, rho,
            beta0, beta1, beta2, tau,
        )
        payoff = discount * notional * torch.clamp(
            1.0 - (1.0 + accrual * strike) * bond, min=0.0
        )
        outputs.extend(price_summary(payoff))
    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}


def price_batch(rows, model_by_id, curve_by_id, product_by_id, *, num_paths, device,
                batch_rows=DEFAULT_BATCH_ROWS, **_: Any):
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(min(batch_rows, len(rows)), num_paths, 4), dtype=torch.float64)
        _price_batch_impl(
            rows[:batch_rows], model_by_id, curve_by_id, product_by_id,
            num_paths=num_paths, target=target, batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows, model_by_id, curve_by_id, product_by_id,
        num_paths=num_paths, target=target, batch_rows=batch_rows,
    )
