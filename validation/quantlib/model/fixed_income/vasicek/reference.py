"""QuantLib adapter for the standalone Vasicek short-rate model."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    _: Mapping[str, Any],
) -> ql.Vasicek:
    """Map one workbench Vasicek row to QuantLib's analytical model."""

    if curve is not None:
        raise ValueError("Standalone Vasicek prices must not reference a curve dataset.")
    context = "Vasicek model"
    return ql.Vasicek(
        finite_number(model, "initial_state", context),
        positive_number(model, "mean_reversion", context),
        finite_number(model, "long_term_mean", context),
        positive_number(model, "volatility", context),
        0.0,
    )
