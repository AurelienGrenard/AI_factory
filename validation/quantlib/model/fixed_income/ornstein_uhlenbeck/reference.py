"""QuantLib Vasicek adapter for the standalone OU short-rate model."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.bermudan_swaption import (
    PreparedBermudanModel,
    bermudan_swaption_times,
    short_exercise_engine,
)
from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.term_structure import discount_curve


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    _: Mapping[str, Any],
) -> ql.Vasicek:
    """Map centered OU parameters to Vasicek with zero long-run mean."""

    if curve is not None:
        raise ValueError("Standalone OU prices must not reference a curve dataset.")
    context = "Ornstein-Uhlenbeck model"
    return ql.Vasicek(
        finite_number(model, "initial_state", context),
        positive_number(model, "mean_reversion", context),
        0.0,
        positive_number(model, "volatility", context),
        0.0,
    )


def quantlib_bermudan_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> PreparedBermudanModel:
    """Map centered OU to the equivalent curve-fitted Hull-White model."""

    vasicek = quantlib_model(model, curve, product)
    initial_state = finite_number(model, "initial_state", "OU model")
    term_structure = discount_curve(
        lambda maturity: vasicek.discountBond(
            0.0, maturity, initial_state
        ),
        bermudan_swaption_times(product),
    )
    reference = ql.HullWhite(
        term_structure,
        positive_number(model, "mean_reversion", "OU model"),
        positive_number(model, "volatility", "OU model"),
    )
    return PreparedBermudanModel(
        reference,
        term_structure,
        short_exercise_engine(product, "fd_hull_white"),
    )
