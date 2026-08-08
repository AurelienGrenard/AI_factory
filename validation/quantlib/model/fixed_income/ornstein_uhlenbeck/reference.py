"""QuantLib Vasicek adapter for the standalone OU short-rate model."""

from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number


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
