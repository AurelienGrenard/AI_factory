"""QuantLib adapter for Hull-White fitted to one Svensson curve."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import positive_number
from validation.quantlib.rate_option import bond_option_times
from validation.quantlib.term_structure import discount_curve, svensson_discount


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> ql.HullWhite:
    """Build the exact-node curve and its Hull-White QuantLib model."""

    if curve is None:
        raise ValueError("Hull-White validation requires a curve dataset.")
    times = bond_option_times(product)
    term_structure = discount_curve(
        lambda maturity: svensson_discount(curve, maturity), times
    )
    return ql.HullWhite(
        term_structure,
        positive_number(model, "mean_reversion", "Hull-White model"),
        positive_number(model, "volatility", "Hull-White model"),
    )
