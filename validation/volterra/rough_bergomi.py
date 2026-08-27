"""Independent Gaussian and direct-hybrid rough-Bergomi references."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import hyp2f1

from validation.volterra.common import (
    MonteCarloEstimate,
    black_forward_option_price,
    estimate_from_antithetic_pairs,
    validate_option_inputs,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoughBergomiParameters:
    spot: float
    risk_free_rate: float
    dividend_yield: float
    xi_0: float
    eta: float
    hurst_exponent: float
    rho: float

    def validate(self) -> None:
        validate_option_inputs(
            self.spot,
            self.spot,
            1.0,
            self.risk_free_rate,
            self.dividend_yield,
        )
        values = (self.xi_0, self.eta, self.hurst_exponent, self.rho)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rough-Bergomi parameters must be finite.")
        if self.xi_0 <= 0.0 or self.eta < 0.0:
            raise ValueError("xi_0 must be positive and eta non-negative.")
        if not 0.0 < self.hurst_exponent < 0.5:
            raise ValueError("hurst_exponent must lie in (0, 0.5).")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1].")


@dataclass(frozen=True)
class GaussianGrid:
    """Cholesky factor of ``(delta W, normalized Volterra driver)``."""

    time_step_count: int
    maturity: float
    covariance: FloatArray
    cholesky: FloatArray
    diagonal_jitter: float


def normalized_driver_covariance(
    first_time: float,
    second_time: float,
    hurst_exponent: float,
) -> float:
    """Covariance of sqrt(2H) int (t-s)^(H-1/2) dW_s."""

    if min(first_time, second_time) < 0.0:
        raise ValueError("times must be non-negative.")
    earlier = min(first_time, second_time)
    later = max(first_time, second_time)
    if earlier == 0.0:
        return 0.0
    if earlier == later:
        return earlier ** (2.0 * hurst_exponent)
    alpha = hurst_exponent - 0.5
    gap = later - earlier
    integral = (
        gap**alpha
        * earlier ** (alpha + 1.0)
        / (alpha + 1.0)
        * hyp2f1(
            -alpha,
            alpha + 1.0,
            alpha + 2.0,
            -earlier / gap,
        )
    )
    return float(2.0 * hurst_exponent * integral)


def driver_increment_covariance(
    driver_time: float,
    interval_start: float,
    interval_end: float,
    hurst_exponent: float,
) -> float:
    """Covariance between the normalized driver and one Brownian increment."""

    upper = min(driver_time, interval_end)
    if upper <= interval_start:
        return 0.0
    exponent = hurst_exponent + 0.5
    return float(
        math.sqrt(2.0 * hurst_exponent)
        * (
            (driver_time - interval_start) ** exponent
            - (driver_time - upper) ** exponent
        )
        / exponent
    )


def gaussian_grid(
    hurst_exponent: float,
    maturity: float,
    time_step_count: int,
) -> GaussianGrid:
    """Build the exact finite-grid Gaussian law in FP64."""

    if not 0.0 < hurst_exponent < 0.5:
        raise ValueError("hurst_exponent must lie in (0, 0.5).")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if time_step_count < 1:
        raise ValueError("time_step_count must be positive.")
    dt = maturity / time_step_count
    dimension = 2 * time_step_count
    covariance = np.empty((dimension, dimension), dtype=np.float64)
    covariance[:time_step_count, :time_step_count] = np.eye(
        time_step_count, dtype=np.float64
    ) * dt
    times = dt * np.arange(1, time_step_count + 1, dtype=np.float64)

    cross = np.empty((time_step_count, time_step_count), dtype=np.float64)
    for driver_index, driver_time in enumerate(times):
        for increment_index in range(time_step_count):
            cross[increment_index, driver_index] = (
                driver_increment_covariance(
                    driver_time,
                    increment_index * dt,
                    (increment_index + 1) * dt,
                    hurst_exponent,
                )
            )
    covariance[:time_step_count, time_step_count:] = cross
    covariance[time_step_count:, :time_step_count] = cross.T
    for row, first_time in enumerate(times):
        for column in range(row + 1):
            value = normalized_driver_covariance(
                first_time, times[column], hurst_exponent
            )
            covariance[time_step_count + row, time_step_count + column] = value
            covariance[time_step_count + column, time_step_count + row] = value

    scale = float(np.max(np.diag(covariance)))
    jitter = 0.0
    identity = np.eye(dimension, dtype=np.float64)
    for relative_jitter in (0.0, 1.0e-15, 1.0e-14, 1.0e-13, 1.0e-12):
        jitter = relative_jitter * scale
        try:
            cholesky = np.linalg.cholesky(covariance + jitter * identity)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        raise RuntimeError(
            "Joint rough-Bergomi covariance is not numerically positive "
            f"definite; minimum eigenvalue={minimum_eigenvalue:.6g}."
        )
    return GaussianGrid(
        time_step_count=time_step_count,
        maturity=maturity,
        covariance=covariance,
        cholesky=cholesky,
        diagonal_jitter=jitter,
    )


def _conditional_payoffs(
    parameters: RoughBergomiParameters,
    strike: float,
    maturity: float,
    brownian_increments: FloatArray,
    variances: FloatArray,
    option_side: str,
) -> FloatArray:
    step_count = brownian_increments.shape[1]
    if variances.shape != (brownian_increments.shape[0], step_count + 1):
        raise ValueError("variances must contain the initial and every grid value.")
    dt = maturity / step_count
    left_variances = variances[:, :-1]
    integrated_variance = dt * np.sum(left_variances, axis=1)
    correlated_integral = np.sum(
        np.sqrt(np.maximum(left_variances, 0.0)) * brownian_increments,
        axis=1,
    )
    conditional_forward = parameters.spot * np.exp(
        (parameters.risk_free_rate - parameters.dividend_yield) * maturity
        + parameters.rho * correlated_integral
        - 0.5 * parameters.rho**2 * integrated_variance
    )
    conditional_log_variance = (
        1.0 - parameters.rho**2
    ) * integrated_variance
    return black_forward_option_price(
        conditional_forward,
        strike,
        math.exp(-parameters.risk_free_rate * maturity),
        conditional_log_variance,
        option_side,
    )


def _exact_grid_variances(
    parameters: RoughBergomiParameters,
    normalized_driver: FloatArray,
    maturity: float,
) -> FloatArray:
    path_count, step_count = normalized_driver.shape
    times = maturity / step_count * np.arange(1, step_count + 1)
    variances = np.empty((path_count, step_count + 1), dtype=np.float64)
    variances[:, 0] = parameters.xi_0
    variances[:, 1:] = parameters.xi_0 * np.exp(
        parameters.eta * normalized_driver
        - 0.5 * parameters.eta**2 * times ** (2.0 * parameters.hurst_exponent)
    )
    return variances


def exact_gaussian_european_option_price(
    parameters: RoughBergomiParameters,
    strike: float,
    maturity: float,
    option_side: str,
    time_step_count: int,
    antithetic_pair_count: int,
    seed: int,
    batch_pair_count: int = 4096,
) -> MonteCarloEstimate:
    """Price from an exact grid Gaussian law and conditional stock payoff."""

    parameters.validate()
    validate_option_inputs(
        parameters.spot,
        strike,
        maturity,
        parameters.risk_free_rate,
        parameters.dividend_yield,
    )
    if antithetic_pair_count < 2 or batch_pair_count < 1:
        raise ValueError("At least two pairs and a positive batch size are required.")
    grid = gaussian_grid(
        parameters.hurst_exponent, maturity, time_step_count
    )
    generator = np.random.default_rng(seed)
    positive_parts: list[FloatArray] = []
    negative_parts: list[FloatArray] = []
    remaining = antithetic_pair_count
    while remaining:
        batch = min(remaining, batch_pair_count)
        normals = generator.standard_normal(
            (batch, 2 * time_step_count), dtype=np.float64
        )
        gaussian = normals @ grid.cholesky.T
        increments = gaussian[:, :time_step_count]
        driver = gaussian[:, time_step_count:]
        positive_parts.append(
            _conditional_payoffs(
                parameters,
                strike,
                maturity,
                increments,
                _exact_grid_variances(parameters, driver, maturity),
                option_side,
            )
        )
        negative_parts.append(
            _conditional_payoffs(
                parameters,
                strike,
                maturity,
                -increments,
                _exact_grid_variances(parameters, -driver, maturity),
                option_side,
            )
        )
        remaining -= batch
    return estimate_from_antithetic_pairs(
        np.concatenate(positive_parts), np.concatenate(negative_parts)
    )


def hybrid_far_weight_matrix(
    hurst_exponent: float,
    maturity: float,
    time_step_count: int,
) -> FloatArray:
    """Dense direct-convolution matrix for the kappa=1 hybrid scheme."""

    alpha = hurst_exponent - 0.5
    alpha_plus_one = alpha + 1.0
    dt = maturity / time_step_count
    matrix = np.zeros((time_step_count, time_step_count), dtype=np.float64)
    for step in range(1, time_step_count):
        for previous in range(step):
            lag = step - previous + 1
            matrix[step, previous] = dt**alpha * (
                lag**alpha_plus_one - (lag - 1) ** alpha_plus_one
            ) / alpha_plus_one
    return matrix


def hybrid_normalized_driver_direct(
    hurst_exponent: float,
    maturity: float,
    brownian_increments: ArrayLike,
    rough_normals: ArrayLike,
    singular_normals: ArrayLike,
) -> FloatArray:
    """Direct O(N^2) normalized hybrid driver at every positive grid time."""

    increments = np.atleast_2d(np.asarray(brownian_increments, dtype=np.float64))
    rough = np.atleast_2d(np.asarray(rough_normals, dtype=np.float64))
    singular = np.atleast_2d(np.asarray(singular_normals, dtype=np.float64))
    if increments.shape != rough.shape or rough.shape != singular.shape:
        raise ValueError("Hybrid random arrays must have identical shapes.")
    step_count = increments.shape[1]
    dt = maturity / step_count
    alpha_plus_one = hurst_exponent + 0.5
    driver_loading = dt**hurst_exponent / alpha_plus_one
    singular_variance = dt ** (2.0 * hurst_exponent) / (
        2.0 * hurst_exponent
    )
    independent_loading = math.sqrt(
        max(singular_variance - driver_loading**2, 0.0)
    )
    local = driver_loading * rough + independent_loading * singular
    far = increments @ hybrid_far_weight_matrix(
        hurst_exponent, maturity, step_count
    ).T
    return math.sqrt(2.0 * hurst_exponent) * (local + far)


def hybrid_normalized_driver_fft(
    hurst_exponent: float,
    maturity: float,
    brownian_increments: ArrayLike,
    rough_normals: ArrayLike,
    singular_normals: ArrayLike,
) -> FloatArray:
    """Independent NumPy FFT evaluation of the same padded convolution."""

    increments = np.atleast_2d(np.asarray(brownian_increments, dtype=np.float64))
    rough = np.atleast_2d(np.asarray(rough_normals, dtype=np.float64))
    singular = np.atleast_2d(np.asarray(singular_normals, dtype=np.float64))
    if increments.shape != rough.shape or rough.shape != singular.shape:
        raise ValueError("Hybrid random arrays must have identical shapes.")
    step_count = increments.shape[1]
    dt = maturity / step_count
    alpha = hurst_exponent - 0.5
    alpha_plus_one = alpha + 1.0
    weights = np.zeros(step_count, dtype=np.float64)
    if step_count > 1:
        lags = np.arange(2, step_count + 1, dtype=np.float64)
        weights[: step_count - 1] = dt**alpha * (
            lags**alpha_plus_one - (lags - 1.0) ** alpha_plus_one
        ) / alpha_plus_one
    transform_length = 1 << (2 * step_count - 1).bit_length()
    convolution = np.fft.irfft(
        np.fft.rfft(increments, transform_length, axis=1)
        * np.fft.rfft(weights, transform_length)[None, :],
        transform_length,
        axis=1,
    )
    far = np.zeros_like(increments)
    if step_count > 1:
        far[:, 1:] = convolution[:, : step_count - 1]
    local_scale = dt**hurst_exponent / alpha_plus_one
    local_variance = dt ** (2.0 * hurst_exponent) / (
        2.0 * hurst_exponent
    )
    independent_scale = math.sqrt(max(local_variance - local_scale**2, 0.0))
    return math.sqrt(2.0 * hurst_exponent) * (
        local_scale * rough + independent_scale * singular + far
    )


def _hybrid_variances(
    parameters: RoughBergomiParameters,
    driver: FloatArray,
    maturity: float,
) -> FloatArray:
    path_count, step_count = driver.shape
    times = maturity / step_count * np.arange(1, step_count + 1)
    variances = np.empty((path_count, step_count + 1), dtype=np.float64)
    variances[:, 0] = parameters.xi_0
    variances[:, 1:] = parameters.xi_0 * np.exp(
        parameters.eta * driver
        - 0.5 * parameters.eta**2 * times ** (2.0 * parameters.hurst_exponent)
    )
    return variances


def hybrid_european_option_price(
    parameters: RoughBergomiParameters,
    strike: float,
    maturity: float,
    option_side: str,
    time_step_count: int,
    antithetic_pair_count: int,
    seed: int,
    batch_pair_count: int = 4096,
    convolution: Literal["direct", "fft"] = "direct",
) -> MonteCarloEstimate:
    """Conditional reference using either direct or NumPy-FFT hybrid weights."""

    parameters.validate()
    validate_option_inputs(
        parameters.spot,
        strike,
        maturity,
        parameters.risk_free_rate,
        parameters.dividend_yield,
    )
    if antithetic_pair_count < 2 or batch_pair_count < 1:
        raise ValueError("At least two pairs and a positive batch size are required.")
    if convolution not in {"direct", "fft"}:
        raise ValueError("convolution must be 'direct' or 'fft'.")
    driver_function = (
        hybrid_normalized_driver_direct
        if convolution == "direct"
        else hybrid_normalized_driver_fft
    )
    generator = np.random.default_rng(seed)
    positive_parts: list[FloatArray] = []
    negative_parts: list[FloatArray] = []
    remaining = antithetic_pair_count
    dt = maturity / time_step_count
    while remaining:
        batch = min(remaining, batch_pair_count)
        rough = generator.standard_normal(
            (batch, time_step_count), dtype=np.float64
        )
        singular = generator.standard_normal(
            (batch, time_step_count), dtype=np.float64
        )
        increments = math.sqrt(dt) * rough
        driver = driver_function(
            parameters.hurst_exponent,
            maturity,
            increments,
            rough,
            singular,
        )
        positive_parts.append(
            _conditional_payoffs(
                parameters,
                strike,
                maturity,
                increments,
                _hybrid_variances(parameters, driver, maturity),
                option_side,
            )
        )
        negative_parts.append(
            _conditional_payoffs(
                parameters,
                strike,
                maturity,
                -increments,
                _hybrid_variances(parameters, -driver, maturity),
                option_side,
            )
        )
        remaining -= batch
    return estimate_from_antithetic_pairs(
        np.concatenate(positive_parts), np.concatenate(negative_parts)
    )


__all__ = (
    "GaussianGrid",
    "RoughBergomiParameters",
    "driver_increment_covariance",
    "exact_gaussian_european_option_price",
    "gaussian_grid",
    "hybrid_european_option_price",
    "hybrid_far_weight_matrix",
    "hybrid_normalized_driver_direct",
    "hybrid_normalized_driver_fft",
    "normalized_driver_covariance",
)
