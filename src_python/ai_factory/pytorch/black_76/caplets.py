"""Vectorized shifted Black-76 caplet pricing."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import (
    nelson_siegel_discount,
    shifted_black_option,
)


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    curve_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    device: str,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(len(rows),), dtype=torch.float64)
    started = perf_counter()
    models = [model_by_id[row["model_id"]] for row in rows]
    curves = [curve_by_id[row["curve_id"]] for row in rows]
    products = [product_by_id[row["product_id"]] for row in rows]
    tensor = lambda values: torch.tensor(values, device=target, dtype=torch.float64)
    volatility = tensor([model["volatility"] for model in models])
    displacement = tensor([model["displacement"] for model in models])
    beta0 = tensor([curve["beta0"] for curve in curves])
    beta1 = tensor([curve["beta1"] for curve in curves])
    beta2 = tensor([curve["beta2"] for curve in curves])
    tau = tensor([curve["tau"] for curve in curves])
    fixing = tensor([product["fixing_time"] for product in products])
    accrual = tensor([product["accrual_period"] for product in products])
    strike = tensor([product["strike"] for product in products])
    notional = tensor([product["notional"] for product in products])
    discount_fixing = nelson_siegel_discount(
        fixing, beta0, beta1, beta2, tau
    )
    discount_payment = nelson_siegel_discount(
        fixing + accrual, beta0, beta1, beta2, tau
    )
    forward = (discount_fixing / discount_payment - 1.0) / accrual
    option = shifted_black_option(
        forward,
        strike,
        displacement,
        volatility * torch.sqrt(fixing),
        torch.ones_like(forward),
    )
    prices = notional * accrual * discount_payment * option
    synchronize(target)
    outputs = [
        {"price": float(price), "standard_error": 0.0}
        for price in prices.cpu().tolist()
    ]
    return outputs, {"wall_seconds": perf_counter() - started}
