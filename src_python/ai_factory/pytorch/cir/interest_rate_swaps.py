"""Vectorized CIR initial swap values."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import cir_bond_price


def _price_batch_impl(rows, model_by_id, product_by_id, target):
    models = [model_by_id[row["model_id"]] for row in rows]
    products = [product_by_id[row["product_id"]] for row in rows]
    host_parameters = torch.tensor(
        [
            (
                model["initial_rate"],
                model["kappa"],
                model["theta"],
                model["volatility"],
                product["start_time"],
                product["accrual_period"],
                product["payment_count"],
                product["fixed_rate"],
                product["notional"],
                product["direction"],
            )
            for model, product in zip(models, products, strict=True)
        ],
        dtype=torch.float64,
    )

    started = perf_counter()
    parameters = host_parameters.to(target)
    (
        initial_rate,
        kappa,
        theta,
        volatility,
        start,
        accrual,
        payment_count,
        fixed_rate,
        notional,
        direction,
    ) = (column[:, None] for column in parameters.unbind(dim=1))
    payment_axis = torch.arange(1, 21, device=target)[None, :]
    payment_dates = start + payment_axis * accrual
    active = payment_axis <= payment_count
    discounts = cir_bond_price(
        initial_rate, kappa, theta, volatility, payment_dates
    )
    annuity = (active * accrual * discounts).sum(1, keepdim=True)
    end = start + payment_count * accrual
    value = notional * direction * (
        cir_bond_price(initial_rate, kappa, theta, volatility, start)
        - cir_bond_price(initial_rate, kappa, theta, volatility, end)
        - fixed_rate * annuity
    )
    host_value = value[:, 0].cpu()
    synchronize(target)
    wall_seconds = perf_counter() - started
    return [
        {"price": float(item), "standard_error": 0.0}
        for item in host_value.tolist()
    ], {"wall_seconds": wall_seconds}


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    device: str,
    **_: Any,
):
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(len(rows), 20), dtype=torch.float64)
        _price_batch_impl(rows, model_by_id, product_by_id, target)
    return _price_batch_impl(rows, model_by_id, product_by_id, target)
