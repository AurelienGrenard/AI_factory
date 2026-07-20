"""Vectorized shifted Black-76 European swaption pricing."""

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
        warmup(target, shape=(len(rows), 20), dtype=torch.float64)
    started = perf_counter()
    models = [model_by_id[row["model_id"]] for row in rows]
    curves = [curve_by_id[row["curve_id"]] for row in rows]
    products = [product_by_id[row["product_id"]] for row in rows]
    tensor = lambda values: torch.tensor(values, device=target, dtype=torch.float64)
    column = lambda values: tensor(values)[:, None]
    volatility = tensor([model["volatility"] for model in models])
    displacement = tensor([model["displacement"] for model in models])
    beta0 = column([curve["beta0"] for curve in curves])
    beta1 = column([curve["beta1"] for curve in curves])
    beta2 = column([curve["beta2"] for curve in curves])
    tau = column([curve["tau"] for curve in curves])
    expiry = tensor([product["expiry"] for product in products])
    accrual = tensor([product["accrual_period"] for product in products])
    strike = tensor([product["fixed_rate"] for product in products])
    notional = tensor([product["notional"] for product in products])
    direction = tensor([product["direction"] for product in products])
    counts = tensor([product["payment_count"] for product in products]).to(torch.int64)
    payments = torch.arange(1, 21, device=target, dtype=torch.float64)[None, :]
    maturities = expiry[:, None] + accrual[:, None] * payments
    discounts = nelson_siegel_discount(maturities, beta0, beta1, beta2, tau)
    active = payments <= counts[:, None]
    annuity = (accrual[:, None] * discounts * active).sum(dim=1)
    discount_start = nelson_siegel_discount(
        expiry, beta0[:, 0], beta1[:, 0], beta2[:, 0], tau[:, 0]
    )
    discount_end = torch.gather(
        discounts, 1, (counts - 1)[:, None]
    )[:, 0]
    forward = (discount_start - discount_end) / annuity
    option = shifted_black_option(
        forward,
        strike,
        displacement,
        volatility * torch.sqrt(expiry),
        direction,
    )
    prices = notional * annuity * option
    synchronize(target)
    outputs = [
        {"price": float(price), "standard_error": 0.0}
        for price in prices.cpu().tolist()
    ]
    return outputs, {"wall_seconds": perf_counter() - started}
