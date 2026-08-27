"""Independent resolvent checks for rough Stein--Stein."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gamma

from validation.volterra.common import MonteCarloEstimate, estimate_from_antithetic_pairs


FloatArray = NDArray[np.float64]

REFERENCE_PAPER = "https://arxiv.org/abs/2009.01219"


@dataclass(frozen=True)
class RoughSteinSteinParameters:
    volatility_level: float
    mean_reversion: float
    volatility_of_volatility: float
    hurst_exponent: float
    spot: float = 1.0
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    rho: float = 0.0


def mittag_leffler_alpha_alpha(alpha: float, argument: float) -> float:
    x = -argument
    crossover = 3.5 + 12.0 * (alpha - 0.5)
    if x > crossover:
        transformed_time = x ** (1.0 / alpha)
        sine_scale = math.sin(math.pi * alpha) / math.pi
        cosine = math.cos(math.pi * alpha)

        def integrand(y: float) -> float:
            r = y / transformed_time
            r_to_alpha = r**alpha
            density = sine_scale * r ** (alpha - 1.0) / (
                r_to_alpha**2 + 2.0 * r_to_alpha * cosine + 1.0
            )
            return y * math.exp(-y) * density

        integral, _ = quad(
            integrand,
            0.0,
            math.inf,
            epsabs=2.0e-14,
            epsrel=2.0e-13,
            limit=400,
        )
        return x ** (1.0 / alpha - 1.0) / transformed_time**2 * integral
    total = 0.0
    power = 1.0
    for order in range(160):
        term = power / gamma(alpha * order + alpha)
        total += term
        if order > 10 and abs(term) < 2.0e-15 * max(abs(total), 1.0):
            break
        power *= argument
    return float(total)


def fractional_resolvent(
    time: float,
    parameters: RoughSteinSteinParameters,
) -> float:
    alpha = parameters.hurst_exponent + 0.5
    laplace_scale = math.sqrt(2.0 * parameters.hurst_exponent) * gamma(alpha)
    return float(
        laplace_scale
        * time ** (alpha - 1.0)
        * mittag_leffler_alpha_alpha(
            alpha,
            -parameters.mean_reversion * laplace_scale * time**alpha,
        )
    )


def resolvent_equation_residual(
    time: float,
    parameters: RoughSteinSteinParameters,
) -> float:
    h = parameters.hurst_exponent
    base = math.sqrt(2.0 * h) * time ** (h - 0.5)
    convolution, _ = quad(
        lambda s: math.sqrt(2.0 * h)
        * (time - s) ** (h - 0.5)
        * fractional_resolvent(s, parameters),
        0.0,
        time,
        points=[0.0, time],
        limit=500,
    )
    return fractional_resolvent(time, parameters) - (
        base - parameters.mean_reversion * convolution
    )


def gaussian_driver(
    parameters: RoughSteinSteinParameters,
    maturity: float,
    increments: FloatArray,
    method: str,
) -> FloatArray:
    increments = np.atleast_2d(np.asarray(increments, dtype=np.float64))
    step_count = increments.shape[1]
    dt = maturity / step_count
    weights = np.empty(step_count, dtype=np.float64)
    for lag in range(1, step_count + 1):
        value, _ = quad(
            lambda t: fractional_resolvent(t, parameters),
            (lag - 1) * dt,
            lag * dt,
            points=[(lag - 1) * dt],
            limit=300,
        )
        weights[lag - 1] = value / dt
    if method == "direct":
        result = np.zeros_like(increments)
        for step in range(step_count):
            result[:, step] = increments[:, : step + 1] @ weights[: step + 1][::-1]
        return result
    if method == "fft":
        length = 1 << (2 * step_count - 1).bit_length()
        return np.fft.irfft(
            np.fft.rfft(increments, length, axis=1)
            * np.fft.rfft(weights, length)[None, :],
            length,
            axis=1,
        )[:, :step_count]
    raise ValueError("method must be direct or fft")


def _resolvent_power_integral(
    parameters: RoughSteinSteinParameters,
    upper: float,
    power: int,
) -> float:
    if upper <= 0.0:
        return 0.0
    lower_x = -math.log(upper)
    cutoff = 70.0
    value = 0.0
    if lower_x < cutoff:
        value, _ = quad(
            lambda x: fractional_resolvent(math.exp(-x), parameters) ** power
            * math.exp(-x),
            lower_x,
            cutoff,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
            limit=400,
        )
    h = parameters.hurst_exponent
    leading = math.sqrt(2.0 * h) ** power
    rate = h + 0.5 if power == 1 else 2.0 * h
    tail_start = max(lower_x, cutoff)
    return float(value + leading * math.exp(-rate * tail_start) / rate)


def hybrid_gaussian_driver(
    parameters: RoughSteinSteinParameters,
    maturity: float,
    rough_normals: ArrayLike,
    singular_normals: ArrayLike,
    method: str,
) -> FloatArray:
    """Production hybrid discretization of the Gaussian resolvent driver."""

    rough = np.atleast_2d(np.asarray(rough_normals, dtype=np.float64))
    singular = np.atleast_2d(np.asarray(singular_normals, dtype=np.float64))
    if rough.shape != singular.shape:
        raise ValueError("normal arrays must have identical shapes")
    step_count = rough.shape[1]
    dt = maturity / step_count
    first_moment = _resolvent_power_integral(parameters, dt, 1)
    second_moment = _resolvent_power_integral(parameters, dt, 2)
    rough_loading = first_moment / math.sqrt(dt)
    singular_loading = math.sqrt(max(second_moment - rough_loading**2, 0.0))
    weights = np.zeros(step_count, dtype=np.float64)
    for index in range(step_count - 1):
        lag = index + 2
        integral, _ = quad(
            lambda time: fractional_resolvent(time, parameters),
            (lag - 1) * dt,
            lag * dt,
            epsabs=2.0e-13,
            epsrel=2.0e-13,
            limit=300,
        )
        weights[index] = integral / dt
    increments = math.sqrt(dt) * rough
    if method == "direct":
        far = np.zeros_like(rough)
        for step in range(1, step_count):
            far[:, step] = increments[:, :step] @ weights[:step][::-1]
    elif method == "fft":
        length = 1 << (2 * step_count - 1).bit_length()
        convolution = np.fft.irfft(
            np.fft.rfft(increments, length, axis=1)
            * np.fft.rfft(weights, length)[None, :],
            length,
            axis=1,
        )[:, :step_count]
        far = np.zeros_like(rough)
        far[:, 1:] = convolution[:, : step_count - 1]
    else:
        raise ValueError("method must be direct or fft")
    return far + rough_loading * rough + singular_loading * singular


def european_option_price(
    parameters: RoughSteinSteinParameters,
    strike: float,
    maturity: float,
    side: str,
    step_count: int,
    antithetic_pair_count: int,
    seed: int,
) -> MonteCarloEstimate:
    """Independent hybrid reference for a rough Stein--Stein vanilla."""

    if side not in {"call", "put"}:
        raise ValueError("side must be call or put")
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((3, antithetic_pair_count, step_count))
    dt = maturity / step_count
    sqrt_dt = math.sqrt(dt)
    residual = math.sqrt(max(1.0 - parameters.rho**2, 0.0))

    def payoffs(signed: FloatArray) -> FloatArray:
        driver = hybrid_gaussian_driver(
            parameters, maturity, signed[0], signed[1], "fft"
        )
        log_spot = np.full(antithetic_pair_count, math.log(parameters.spot))
        volatility = np.full(
            antithetic_pair_count, parameters.volatility_level
        )
        for step in range(step_count):
            stock_normal = (
                parameters.rho * signed[0, :, step]
                + residual * signed[2, :, step]
            )
            log_spot += (
                (parameters.risk_free_rate - parameters.dividend_yield) * dt
                - 0.5 * volatility**2 * dt
                + volatility * sqrt_dt * stock_normal
            )
            volatility = parameters.volatility_level
            volatility += parameters.volatility_of_volatility * driver[:, step]
        terminal = np.exp(log_spot)
        intrinsic = np.maximum(
            terminal - strike if side == "call" else strike - terminal,
            0.0,
        )
        return math.exp(-parameters.risk_free_rate * maturity) * intrinsic

    return estimate_from_antithetic_pairs(payoffs(noise), payoffs(-noise))


__all__ = (
    "REFERENCE_PAPER",
    "RoughSteinSteinParameters",
    "european_option_price",
    "fractional_resolvent",
    "gaussian_driver",
    "hybrid_gaussian_driver",
    "resolvent_equation_residual",
)
