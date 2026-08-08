"""Adapt standalone G2 to QuantLib G2 through its endogenous initial curve."""

import math
from typing import Any, Mapping

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.rate_option import bond_option_times
from validation.quantlib.term_structure import discount_curve


def _loading(mean_reversion: float, maturity: float) -> float:
    """Return the exact OU integral loading B(0,T)."""

    return -math.expm1(-mean_reversion * maturity) / mean_reversion


def _integral_variance(
    mean_reversion: float, volatility: float, maturity: float
) -> float:
    """Return the variance of one centered OU time integral."""

    loading = _loading(mean_reversion, maturity)
    second_loading = -math.expm1(-2.0 * mean_reversion * maturity) / (
        2.0 * mean_reversion
    )
    return volatility * volatility / (mean_reversion * mean_reversion) * (
        maturity - 2.0 * loading + second_loading
    )


def _discount(model: Mapping[str, Any], maturity: float) -> float:
    """Evaluate the initial discount curve implied by raw G2 states."""

    context = "G2 model"
    a = positive_number(model, "mean_reversion_x", context)
    sigma = positive_number(model, "volatility_x", context)
    b = positive_number(model, "mean_reversion_y", context)
    eta = positive_number(model, "volatility_y", context)
    rho = finite_number(model, "correlation", context)
    state_x = finite_number(model, "initial_state_x", context)
    state_y = finite_number(model, "initial_state_y", context)
    cross_variance = rho * sigma * eta / (a * b) * (
        maturity
        - _loading(a, maturity)
        - _loading(b, maturity)
        + _loading(a + b, maturity)
    )
    integral_variance = (
        _integral_variance(a, sigma, maturity)
        + _integral_variance(b, eta, maturity)
        + 2.0 * cross_variance
    )
    return math.exp(
        -_loading(a, maturity) * state_x
        - _loading(b, maturity) * state_y
        + 0.5 * integral_variance
    )


def quantlib_model(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
) -> ql.G2:
    """Fit QuantLib G2++ to the exact initial curve of standalone G2."""

    if curve is not None:
        raise ValueError("Standalone G2 prices must not reference a curve dataset.")
    context = "G2 model"
    a = positive_number(model, "mean_reversion_x", context)
    sigma = positive_number(model, "volatility_x", context)
    b = positive_number(model, "mean_reversion_y", context)
    eta = positive_number(model, "volatility_y", context)
    rho = finite_number(model, "correlation", context)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("G2 model: correlation must lie in [-1, 1].")
    term_structure = discount_curve(
        lambda maturity: _discount(model, maturity), bond_option_times(product)
    )
    return ql.G2(term_structure, a, sigma, b, eta, rho)
