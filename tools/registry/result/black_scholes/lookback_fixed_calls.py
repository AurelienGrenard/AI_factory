"""Black-Scholes fixed-lookback result pipeline facade."""

from __future__ import annotations

from typing import Any

from tools.registry.result.black_scholes.common import (
    AUDIT_ROW_COUNT,
    PRODUCTION_ROW_COUNT,
    generate_result as _generate_result,
)

__all__ = [
    "AUDIT_ROW_COUNT",
    "PRODUCTION_ROW_COUNT",
    "generate_result",
]


def generate_result(**kwargs: Any):
    return _generate_result(product_family="lookback_fixed_calls", **kwargs)
