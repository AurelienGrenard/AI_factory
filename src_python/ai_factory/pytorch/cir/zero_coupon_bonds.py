"""Vectorized CIR time-zero zero-coupon bond values."""

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
                product["notional"],
                model["initial_rate"],
                model["kappa"],
                model["theta"],
                model["volatility"],
                product["maturity"],
            )
            for model, product in zip(models, products, strict=True)
        ],
        dtype=torch.float64,
    )

    started = perf_counter()
    parameters = host_parameters.to(target)
    notional, initial_rate, kappa, theta, volatility, maturity = parameters.unbind(dim=1)
    price = notional * cir_bond_price(
        initial_rate,
        kappa,
        theta,
        volatility,
        maturity,
    )
    host_price = price.cpu()
    synchronize(target)
    wall_seconds = perf_counter() - started
    return [
        {"price": float(value), "standard_error": 0.0}
        for value in host_price.tolist()
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
        warmup(target, shape=(len(rows),), dtype=torch.float64)
        _price_batch_impl(rows, model_by_id, product_by_id, target)
    return _price_batch_impl(rows, model_by_id, product_by_id, target)
