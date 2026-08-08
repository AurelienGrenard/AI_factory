"""Shared validation and QuantLib construction for Black-Scholes rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.term_structure import (
    DAY_COUNTER,
    REFERENCE_DATE,
    flat_curve,
)


@dataclass(frozen=True)
class BlackScholesReference:
    """Validated scalar parameters and their QuantLib process."""

    process: ql.BlackScholesMertonProcess
    spot: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float


def quantlib_reference(parameters: Mapping[str, Any]) -> BlackScholesReference:
    """Map one workbench row to QuantLib's Black-Scholes convention."""

    context = "Black-Scholes model"
    spot = positive_number(parameters, "spot", context)
    risk_free_rate = finite_number(parameters, "risk_free_rate", context)
    dividend_yield = finite_number(parameters, "dividend_yield", context)
    volatility = positive_number(parameters, "volatility", context)
    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        flat_curve(dividend_yield),
        flat_curve(risk_free_rate),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                REFERENCE_DATE,
                ql.NullCalendar(),
                volatility,
                DAY_COUNTER,
            )
        ),
    )
    return BlackScholesReference(
        process, spot, risk_free_rate, dividend_yield, volatility
    )
