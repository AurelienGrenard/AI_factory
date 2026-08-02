"""Small typed readers shared by model-specific QuantLib adapters."""

from __future__ import annotations

import math
from typing import Any, Mapping


def finite_number(
    parameters: Mapping[str, Any], field: str, context: str
) -> float:
    """Return one required finite numeric parameter."""

    value = parameters.get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context}: {field} must be finite and numeric.")
    return float(value)


def positive_number(
    parameters: Mapping[str, Any], field: str, context: str
) -> float:
    """Return one required finite strictly positive parameter."""

    value = finite_number(parameters, field, context)
    if value <= 0.0:
        raise ValueError(f"{context}: {field} must be positive.")
    return value
