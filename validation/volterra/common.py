"""Numerical and statistical primitives shared by Volterra references."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson
from scipy.special import ndtr


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class MonteCarloEstimate:
    """One price estimate and the standard error of its independent samples."""

    price: float
    standard_error: float
    independent_sample_count: int


@dataclass(frozen=True)
class PriceCertification:
    """Explicit four-sigma comparison against one independent reference."""

    passed: bool
    difference: float
    allowance: float
    z_score: float


def _finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def validate_option_inputs(
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> None:
    _finite_positive(spot, "spot")
    _finite_positive(strike, "strike")
    _finite_positive(maturity, "maturity")
    if not math.isfinite(risk_free_rate) or not math.isfinite(dividend_yield):
        raise ValueError("rates must be finite.")


def black_forward_option_price(
    forward: ArrayLike,
    strike: float,
    discount: float,
    log_variance: ArrayLike,
    option_side: str,
) -> FloatArray:
    """Vectorized Black price from a forward and its log-return variance."""

    if option_side not in {"call", "put"}:
        raise ValueError("option_side must be 'call' or 'put'.")
    forwards = np.asarray(forward, dtype=np.float64)
    variances = np.maximum(np.asarray(log_variance, dtype=np.float64), 0.0)
    standard_deviation = np.sqrt(variances)
    intrinsic_call = discount * np.maximum(forwards - strike, 0.0)
    regular = standard_deviation > 1.0e-14
    result = np.array(intrinsic_call, copy=True, ndmin=1)
    if np.any(regular):
        std = standard_deviation[regular]
        selected_forward = np.broadcast_to(forwards, variances.shape)[regular]
        d1 = np.log(selected_forward / strike) / std + 0.5 * std
        d2 = d1 - std
        result[regular] = discount * (
            selected_forward * ndtr(d1) - strike * ndtr(d2)
        )
    if option_side == "put":
        result = result - discount * (
            np.broadcast_to(forwards, variances.shape) - strike
        )
    return result


def estimate_from_antithetic_pairs(
    positive_payoffs: ArrayLike,
    negative_payoffs: ArrayLike,
) -> MonteCarloEstimate:
    """Estimate uncertainty from independent antithetic pair averages."""

    positive = np.asarray(positive_payoffs, dtype=np.float64)
    negative = np.asarray(negative_payoffs, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("Antithetic payoff arrays must be one-dimensional peers.")
    if positive.size < 2:
        raise ValueError("At least two antithetic pairs are required.")
    pair_means = 0.5 * (positive + negative)
    return MonteCarloEstimate(
        price=float(np.mean(pair_means)),
        standard_error=float(
            np.std(pair_means, ddof=1) / math.sqrt(pair_means.size)
        ),
        independent_sample_count=int(pair_means.size),
    )


def certify_price(
    generated_price: float,
    generated_standard_error: float,
    reference_price: float,
    reference_standard_error: float = 0.0,
    numerical_allowance: float = 0.0,
    standard_error_multiplier: float = 4.0,
) -> PriceCertification:
    """Compare two estimators without confusing noise and numerical bias."""

    values = (
        generated_price,
        generated_standard_error,
        reference_price,
        reference_standard_error,
        numerical_allowance,
        standard_error_multiplier,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Certification inputs must be finite.")
    if min(
        generated_standard_error,
        reference_standard_error,
        numerical_allowance,
        standard_error_multiplier,
    ) < 0.0:
        raise ValueError("Errors, allowances and multipliers must be non-negative.")
    combined_error = math.hypot(
        generated_standard_error, reference_standard_error
    )
    difference = generated_price - reference_price
    allowance = numerical_allowance + standard_error_multiplier * combined_error
    z_score = (
        difference / combined_error
        if combined_error > 0.0
        else math.copysign(math.inf, difference) if difference else 0.0
    )
    return PriceCertification(
        passed=abs(difference) <= allowance,
        difference=difference,
        allowance=allowance,
        z_score=z_score,
    )


def lewis_european_option_price(
    log_forward_mgf: Callable[[ComplexArray], ComplexArray],
    spot: float,
    strike: float,
    maturity: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_side: str,
    integration_cutoff: float = 80.0,
    integration_point_count: int = 1601,
) -> float:
    """Lewis inversion using only the half-moment of the log-forward.

    ``log_forward_mgf(z)`` returns ``log E[(S_T/F_T)**z]`` for a vector of
    complex exponents.  A fixed Simpson grid makes convergence comparisons
    deterministic and avoids adaptive quadrature calling a costly Riccati
    solver in an unpredictable order.
    """

    validate_option_inputs(
        spot, strike, maturity, risk_free_rate, dividend_yield
    )
    if option_side not in {"call", "put"}:
        raise ValueError("option_side must be 'call' or 'put'.")
    _finite_positive(integration_cutoff, "integration_cutoff")
    if integration_point_count < 3 or integration_point_count % 2 == 0:
        raise ValueError("integration_point_count must be odd and at least 3.")

    frequencies = np.linspace(
        0.0, integration_cutoff, integration_point_count, dtype=np.float64
    )
    exponents = 0.5 + 1j * frequencies
    log_mgf = np.asarray(log_forward_mgf(exponents), dtype=np.complex128)
    if log_mgf.shape != exponents.shape or not np.all(np.isfinite(log_mgf)):
        raise RuntimeError(
            "The characteristic-function solver returned invalid values."
        )
    forward = spot * math.exp(
        (risk_free_rate - dividend_yield) * maturity
    )
    log_forward_moneyness = math.log(forward / strike)
    integrand = np.real(
        np.exp(1j * frequencies * log_forward_moneyness + log_mgf)
    ) / (frequencies * frequencies + 0.25)
    integral = float(simpson(integrand, x=frequencies))
    discount = math.exp(-risk_free_rate * maturity)
    call = spot * math.exp(-dividend_yield * maturity) - (
        discount * math.sqrt(strike * forward) * integral / math.pi
    )
    if option_side == "call":
        return call
    return call - spot * math.exp(-dividend_yield * maturity) + strike * discount


__all__ = (
    "ComplexArray",
    "FloatArray",
    "MonteCarloEstimate",
    "PriceCertification",
    "black_forward_option_price",
    "certify_price",
    "estimate_from_antithetic_pairs",
    "lewis_european_option_price",
    "validate_option_inputs",
)
