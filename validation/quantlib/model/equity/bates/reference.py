"""Shared QuantLib construction for every Bates product validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.term_structure import REFERENCE_DATE, flat_curve


@dataclass(frozen=True)
class BatesReference:
    """QuantLib process and model built from one validated Bates row."""

    process: ql.BatesProcess
    model: ql.BatesModel
    spot: float


def quantlib_reference(parameters: Mapping[str, Any]) -> BatesReference:
    """Map one workbench Bates row to QuantLib's risk-neutral model."""

    context = "Bates model"
    spot = positive_number(parameters, "spot", context)
    risk_free_rate = finite_number(parameters, "risk_free_rate", context)
    dividend_yield = finite_number(parameters, "dividend_yield", context)
    initial_variance = finite_number(parameters, "initial_variance", context)
    if initial_variance < 0.0:
        raise ValueError(f"{context}: initial_variance must be non-negative.")
    kappa = positive_number(parameters, "kappa", context)
    theta = positive_number(parameters, "theta", context)
    gamma = positive_number(parameters, "gamma", context)
    rho = finite_number(parameters, "rho", context)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"{context}: rho must lie in [-1, 1].")
    jump_intensity = finite_number(parameters, "jump_intensity", context)
    if jump_intensity < 0.0:
        raise ValueError(f"{context}: jump_intensity must be non-negative.")
    jump_log_mean = finite_number(parameters, "jump_log_mean", context)
    jump_log_volatility = finite_number(
        parameters, "jump_log_volatility", context
    )
    if jump_log_volatility < 0.0:
        raise ValueError(
            f"{context}: jump_log_volatility must be non-negative."
        )

    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    process = ql.BatesProcess(
        flat_curve(risk_free_rate),
        flat_curve(dividend_yield),
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        initial_variance,
        kappa,
        theta,
        gamma,
        rho,
        jump_intensity,
        jump_log_mean,
        jump_log_volatility,
    )
    return BatesReference(process, ql.BatesModel(process), spot)
