"""QuantLib Monte-Carlo references for Black-Scholes option datasets."""

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
        "asian_call",
        "asian_put",
        "athena_autocall",
        "cliquet",
        "double_knock_out_call",
        "double_knock_out_put",
        "down_and_in_put",
        "down_and_out_put",
        "lookback_option",
        "phoenix_autocall",
        "phoenix_memory_autocall",
        "up_and_in_call",
        "up_and_out_call",
        "up_no_touch",
        "up_one_touch",
    }
)


def validation_from_quantlib_black_scholes_monte_carlo_option(
    price_dataset_path: str | Path,
    product_kind: str,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
    path_reference_pairs: int = 1024,
) -> PriceValidationReport:
    """Validate a product through its compatible QuantLib MC reference."""

    if product_kind not in SUPPORTED_PRODUCTS:
        raise ValueError(
            f"Black-Scholes product '{product_kind}' has no compatible "
            "QuantLib Monte-Carlo reference."
        )
    return validation_from_quantlib_black_scholes_option(
        price_dataset_path,
        product_kind,
        tolerances,
        regime,
        row_ids,
        path_reference_pairs,
    )
