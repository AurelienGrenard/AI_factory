"""Rough Heston American put pricing with batched Longstaff-Schwartz."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.american_options import price_puts_from_paths_batch
from ai_factory.pytorch.common.pathwise_products import parameter_tensor
from ai_factory.pytorch.rough_heston.pathwise import simulate_state_paths_batch

DEFAULT_BATCH_ROWS = 8


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    _sync(target)
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        paths, factor_paths = simulate_state_paths_batch(
            chunk, model_by_id, product_by_id,
            num_paths, num_steps, target, dtype,
        )
        strikes = parameter_tensor(
            chunk, model_by_id, product_by_id, "strike",
            source="product", device=target, dtype=dtype,
        )
        maturities = parameter_tensor(
            chunk, model_by_id, product_by_id, "maturity",
            source="product", device=target, dtype=dtype,
        )
        rates = parameter_tensor(
            chunk, model_by_id, product_by_id, "risk_free_rate",
            source="model", device=target, dtype=dtype,
        )
        thetas = parameter_tensor(
            chunk, model_by_id, product_by_id, "theta",
            source="model", device=target, dtype=dtype,
        )
        outputs.extend(
            price_puts_from_paths_batch(
                paths,
                strikes=strikes,
                maturities=maturities,
                rates=rates,
                regression_features=factor_paths / thetas[:, None, None, :],
            )
        )
    _sync(target)
    return outputs, {"wall_seconds": perf_counter() - started}
