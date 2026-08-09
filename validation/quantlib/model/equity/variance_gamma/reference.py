"""Shared QuantLib construction for Variance-Gamma validators."""

from dataclasses import dataclass
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.term_structure import REFERENCE_DATE, flat_curve


@dataclass(frozen=True)
class VarianceGammaReference:
    """QuantLib process built independently from one generated model row."""

    process: ql.VarianceGammaProcess
    spot: float


def quantlib_reference(
    parameters: Mapping[str, Any],
) -> VarianceGammaReference:
    """Validate and map one workbench row to QuantLib's VG convention."""

    context = "Variance-Gamma model"
    spot = positive_number(parameters, "spot", context)
    rate = finite_number(parameters, "risk_free_rate", context)
    dividend = finite_number(parameters, "dividend_yield", context)
    sigma = positive_number(parameters, "sigma", context)
    nu = positive_number(parameters, "nu", context)
    theta = finite_number(parameters, "theta", context)
    if 1.0 - theta * nu - 0.5 * sigma * sigma * nu <= 0.0:
        raise ValueError(f"{context}: the unit exponential moment must exist.")

    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    process = ql.VarianceGammaProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        flat_curve(dividend),
        flat_curve(rate),
        sigma,
        nu,
        theta,
    )
    return VarianceGammaReference(process, spot)
