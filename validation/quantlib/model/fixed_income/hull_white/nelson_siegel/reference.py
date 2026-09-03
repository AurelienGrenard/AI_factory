"""QuantLib adapter for Hull-White fitted to one Nelson-Siegel curve."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.bermudan_swaption import (
    PreparedBermudanModel,
    bermudan_swaption_times,
    short_exercise_engine,
)
from validation.quantlib.parameters import positive_number
from validation.quantlib.rate_option import bond_option_times
from validation.quantlib.swaption import swaption_times
from validation.quantlib.term_structure import discount_curve, nelson_siegel_discount


def _required_times(product: Mapping[str, Any]) -> tuple[float, ...]:
    if "first_exercise_time" in product:
        return bermudan_swaption_times(product)
    if "exercise_time" in product:
        return swaption_times(product)
    return bond_option_times(product)


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> ql.HullWhite:
    """Build the exact-node curve and its Hull-White QuantLib model."""

    if curve is None:
        raise ValueError("Hull-White validation requires a curve dataset.")
    times = _required_times(product)
    term_structure = discount_curve(
        lambda maturity: nelson_siegel_discount(curve, maturity), times
    )
    return ql.HullWhite(
        term_structure,
        positive_number(model, "mean_reversion", "Hull-White model"),
        positive_number(model, "volatility", "Hull-White model"),
    )


def quantlib_bermudan_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> PreparedBermudanModel:
    """Build the PDE-ready Hull-White Bermudan reference."""

    reference = quantlib_model(model, curve, product)
    return PreparedBermudanModel(
        reference,
        reference.termStructure(),
        short_exercise_engine(product, "fd_hull_white"),
    )
