"""QuantLib specialized references for Black-Scholes option datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from validation.quantlib.model.equity.black_scholes.equity_option import (
    validation_from_quantlib_black_scholes_option,
)
from validation.quantlib.price_validation import (
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
)


SUPPORTED_PRODUCTS = frozenset(
    {
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "digital_call",
        "digital_put",
        "european_call",
        "european_put",
        "forward_start_call",
        "forward_start_put",
        "gap_call",
        "gap_put",
        "geometric_asian_call",
        "geometric_asian_put",
        "range_accrual",
        "straddle",
    }
)


def validation_from_quantlib_black_scholes_specialized_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Validate a product whose QuantLib reference is non-Monte-Carlo."""

    if product_kind not in SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Black-Scholes product '{product_kind}' has no compatible "
            "QuantLib specialized reference."
        )
    return validation_from_quantlib_black_scholes_option(
        price_dataset_path,
        product_kind,
        tolerances,
        regime,
        row_ids,
    )
