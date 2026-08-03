"""QuantLib adapter for G2++ fitted to one Svensson curve."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.rate_option import bond_option_times
from validation.quantlib.term_structure import discount_curve, svensson_discount


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> ql.G2:
    """Build the exact-node curve and its G2++ QuantLib model."""

    if curve is None:
        raise ValueError("G2++ validation requires a curve dataset.")
    context = "G2++ model"
    correlation = finite_number(model, "correlation", context)
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"{context}: correlation must lie in [-1, 1].")
    term_structure = discount_curve(
        lambda maturity: svensson_discount(curve, maturity),
        bond_option_times(product),
    )
    return ql.G2(
        term_structure,
        positive_number(model, "mean_reversion_x", context),
        positive_number(model, "volatility_x", context),
        positive_number(model, "mean_reversion_y", context),
        positive_number(model, "volatility_y", context),
        correlation,
    )
