"""Vectorized G2++ time-zero zero-coupon bond values."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import nelson_siegel_discount


def _price_batch_impl(
    rows: list[dict[str, Any]],
    curve_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    target: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    curves = [curve_by_id[row["curve_id"]] for row in rows]
    products = [product_by_id[row["product_id"]] for row in rows]
    host_parameters = torch.tensor(
        [
            (
                product["maturity"],
                product["notional"],
                curve["beta0"],
                curve["beta1"],
                curve["beta2"],
                curve["tau"],
            )
            for curve, product in zip(curves, products, strict=True)
        ],
        dtype=torch.float64,
    )

    started = perf_counter()
    parameters = host_parameters.to(target)
    maturity, notional, beta0, beta1, beta2, tau = parameters.unbind(dim=1)
    price = notional * nelson_siegel_discount(
        maturity,
        beta0,
        beta1,
        beta2,
        tau,
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
    curve_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    device: str,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    del model_by_id
    target = resolve_device(device)
    if target.type == "cuda" and rows:
        warmup(target, shape=(len(rows),), dtype=torch.float64)
        _price_batch_impl(rows, curve_by_id, product_by_id, target)
    return _price_batch_impl(rows, curve_by_id, product_by_id, target)
