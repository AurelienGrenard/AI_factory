"""Independent FP64 reference for log-modulated rough Bergomi."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from validation.volterra.common import MonteCarloEstimate, estimate_from_antithetic_pairs


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LogModulatedRoughBergomiParameters:
    spot: float
    risk_free_rate: float
    dividend_yield: float
    xi_0: float
    eta: float
    hurst_exponent: float
    rho: float
    log_modulation_scale: float
    log_modulation_power: float


def _unnormalized_kernel(
    time: float, parameters: LogModulatedRoughBergomiParameters
) -> float:
    if not time > 0.0:
        raise ValueError("The log-modulated kernel is defined for positive time.")
    modulation = max(
        parameters.log_modulation_scale * math.log(1.0 / time), 1.0
    ) ** (-parameters.log_modulation_power)
    return time ** (parameters.hurst_exponent - 0.5) * modulation


def _unnormalized_power_integral(
    upper: float,
    power: float,
    parameters: LogModulatedRoughBergomiParameters,
) -> float:
    """Integrate ``K_unscaled**power`` without evaluating the singularity."""

    if upper <= 0.0:
        return 0.0
    lower_x = -math.log(upper)
    rate = power * (parameters.hurst_exponent - 0.5) + 1.0
    value, _ = quad(
        lambda x: math.exp(-rate * x)
        * max(parameters.log_modulation_scale * x, 1.0)
        ** (-power * parameters.log_modulation_power),
        lower_x,
        math.inf,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
        limit=400,
    )
    return float(value)


def normalization(parameters: LogModulatedRoughBergomiParameters) -> float:
    squared = _unnormalized_power_integral(1.0, 2.0, parameters)
    return 1.0 / math.sqrt(squared)


def kernel(time: float, parameters: LogModulatedRoughBergomiParameters) -> float:
    return normalization(parameters) * _unnormalized_kernel(time, parameters)


def hybrid_driver(
    parameters: LogModulatedRoughBergomiParameters,
    maturity: float,
    rough_normals: ArrayLike,
    singular_normals: ArrayLike,
    method: Literal["direct", "fft"],
) -> FloatArray:
    rough = np.atleast_2d(np.asarray(rough_normals, dtype=np.float64))
    singular = np.atleast_2d(np.asarray(singular_normals, dtype=np.float64))
    if rough.shape != singular.shape:
        raise ValueError("normal arrays must have identical shapes")
    path_count, step_count = rough.shape
    dt = maturity / step_count
    scale = normalization(parameters)
    integral_one = scale * _unnormalized_power_integral(dt, 1.0, parameters)
    integral_two = scale**2 * _unnormalized_power_integral(
        dt, 2.0, parameters
    )
    rough_loading = integral_one / math.sqrt(dt)
    residual_loading = math.sqrt(max(integral_two - rough_loading**2, 0.0))
    weights = np.zeros(step_count, dtype=np.float64)
    for index in range(step_count - 1):
        lag = index + 2
        value, _ = quad(
            lambda x: scale * _unnormalized_kernel(x, parameters),
            (lag - 1) * dt,
            lag * dt,
            limit=100,
        )
        weights[index] = value / dt
    increments = math.sqrt(dt) * rough
    if method == "direct":
        far = np.zeros_like(rough)
        for step in range(1, step_count):
            far[:, step] = increments[:, :step] @ weights[:step][::-1]
    elif method == "fft":
        length = 1 << (2 * step_count - 1).bit_length()
        transformed = np.fft.rfft(increments, length, axis=1)
        convolution = np.fft.irfft(
            transformed * np.fft.rfft(weights, length)[None, :],
            length,
            axis=1,
        )[:, :step_count]
        far = np.zeros_like(rough)
        far[:, 1:] = convolution[:, : step_count - 1]
    else:
        raise ValueError("method must be direct or fft")
    return far + rough_loading * rough + residual_loading * singular


def european_option_price(
    parameters: LogModulatedRoughBergomiParameters,
    strike: float,
    maturity: float,
    side: str,
    step_count: int,
    antithetic_pair_count: int,
    seed: int,
) -> MonteCarloEstimate:
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((3, antithetic_pair_count, step_count))
    scale = normalization(parameters)

    def payoffs(signed: FloatArray) -> FloatArray:
        driver = hybrid_driver(parameters, maturity, signed[0], signed[1], "fft")
        dt = maturity / step_count
        times = dt * np.arange(1, step_count + 1)
        variances = parameters.xi_0 * np.exp(
            parameters.eta * driver
            - 0.5 * parameters.eta**2
            * (
                scale**2
                * np.array([
                    _unnormalized_power_integral(time, 2.0, parameters)
                    for time in times
                ])
            )[None, :]
        )
        log_spot = np.full(antithetic_pair_count, math.log(parameters.spot))
        variance = np.full(antithetic_pair_count, parameters.xi_0)
        residual = math.sqrt(max(1.0 - parameters.rho**2, 0.0))
        for step in range(step_count):
            normal = parameters.rho * signed[0, :, step] + residual * signed[2, :, step]
            log_spot += (
                (parameters.risk_free_rate - parameters.dividend_yield) * dt
                - 0.5 * variance * dt
                + np.sqrt(variance * dt) * normal
            )
            variance = variances[:, step]
        terminal = np.exp(log_spot)
        intrinsic = np.maximum(
            terminal - strike if side == "call" else strike - terminal, 0.0
        )
        return math.exp(-parameters.risk_free_rate * maturity) * intrinsic

    return estimate_from_antithetic_pairs(payoffs(noise), payoffs(-noise))


__all__ = (
    "LogModulatedRoughBergomiParameters",
    "european_option_price",
    "hybrid_driver",
    "kernel",
    "normalization",
)
