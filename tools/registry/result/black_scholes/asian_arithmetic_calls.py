"""Black-Scholes arithmetic-Asian result pipeline facade."""

from __future__ import annotations

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


def generate_result(**kwargs):
    return _generate_result(product_family="asian_arithmetic_calls", **kwargs)
