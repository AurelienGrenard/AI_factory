"""Vectorized G2++ initial swap values."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import nelson_siegel_discount


def _price_batch_impl(rows, curve_by_id, product_by_id, target):
    curves = [curve_by_id[row["curve_id"]] for row in rows]
    products = [product_by_id[row["product_id"]] for row in rows]
    host_parameters = torch.tensor(
        [
            (
                curve["beta0"],
                curve["beta1"],
                curve["beta2"],
                curve["tau"],
                product["start_time"],
                product["accrual_period"],
                product["payment_count"],
                product["fixed_rate"],
                product["notional"],
                product["direction"],
            )
            for curve, product in zip(curves, products, strict=True)
        ],
        dtype=torch.float64,
    )

    started = perf_counter()
    parameters = host_parameters.to(target)
    (
        beta0,
        beta1,
        beta2,
        tau,
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
    discounts = nelson_siegel_discount(
        payment_dates, beta0, beta1, beta2, tau
    )
    annuity = (active * accrual * discounts).sum(1, keepdim=True)
    end = start + payment_count * accrual
    value = notional * direction * (
        nelson_siegel_discount(start, beta0, beta1, beta2, tau)
        - nelson_siegel_discount(end, beta0, beta1, beta2, tau)
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
    curve_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    device: str,
    **_: Any,
):
    del model_by_id
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(len(rows), 20), dtype=torch.float64)
        _price_batch_impl(rows, curve_by_id, product_by_id, target)
    return _price_batch_impl(rows, curve_by_id, product_by_id, target)
