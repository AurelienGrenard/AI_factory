"""Independent path references for restricted quadratic rough Heston.

The equations follow Gatheral, Jusselin and Rosenbaum's restricted QRH model.
The dense convolution below is intentionally independent from the production
factor recurrence.  Supplying the same exponential kernel isolates the lift
implementation; supplying the fractional kernel exposes the kernel-fit error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma

from validation.volterra.rough_heston import ExponentialKernel


REFERENCE_PAPER = "https://arxiv.org/abs/2001.01789"
UPSTREAM_REPOSITORY = "https://github.com/jgatheral/QuadraticRoughHeston"

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticRoughHestonParameters:
    spot: float
    risk_free_rate: float
    dividend_yield: float
    initial_feedback: float
    quadratic_scale: float
    quadratic_shift: float
    variance_floor: float
    feedback_rate: float
    feedback_volatility: float
    hurst_exponent: float

    def variance(self, feedback: FloatArray) -> FloatArray:
        return (
            self.quadratic_scale * (feedback - self.quadratic_shift) ** 2
            + self.variance_floor
        )


@dataclass(frozen=True)
class SimulatedPaths:
    log_spot: FloatArray
    feedback: FloatArray


def fractional_cell_average_weights(
    hurst_exponent: float,
    time_step: float,
    step_count: int,
) -> FloatArray:
    """Cell averages of t**(H-1/2) / Gamma(H+1/2)."""

    alpha = hurst_exponent + 0.5
    lags = np.arange(1, step_count + 1, dtype=np.float64)
    return (
        time_step ** (alpha - 1.0)
        * (lags**alpha - (lags - 1.0) ** alpha)
        / gamma(alpha + 1.0)
    )


def exponential_cell_average_weights(
    kernel: ExponentialKernel,
    time_step: float,
    step_count: int,
) -> FloatArray:
    """Cell averages of an externally supplied exponential kernel."""

    nodes = np.asarray(kernel.nodes, dtype=np.float64)
    weights = np.asarray(kernel.weights, dtype=np.float64)
    lags = np.arange(step_count, dtype=np.float64)
    integrals = np.where(
        nodes > 0.0,
        -np.expm1(-nodes * time_step) / nodes,
        time_step,
    )
    return (
        np.exp(-lags[:, None] * time_step * nodes[None, :])
        @ (weights * integrals)
    ) / time_step


def simulate_dense_convolution(
    parameters: QuadraticRoughHestonParameters,
    maturity: float,
    normals: ArrayLike,
    cell_average_weights: ArrayLike,
) -> SimulatedPaths:
    """O(N^2) balanced-cell reference for an arbitrary Volterra kernel."""

    normal_values = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    path_count, step_count = normal_values.shape
    weights = np.asarray(cell_average_weights, dtype=np.float64)
    if weights.shape != (step_count,):
        raise ValueError("One cell-average kernel weight is required per lag.")
    dt = maturity / step_count
    sqrt_dt = math.sqrt(dt)
    feedback = np.empty((path_count, step_count + 1), dtype=np.float64)
    feedback[:, 0] = parameters.initial_feedback
    forces = np.empty((path_count, step_count), dtype=np.float64)
    log_spot = np.full(path_count, math.log(parameters.spot), dtype=np.float64)
    carry = (parameters.risk_free_rate - parameters.dividend_yield) * dt

    for step in range(step_count):
        current_feedback = feedback[:, step]
        variance = parameters.variance(current_feedback)
        normal = normal_values[:, step]
        log_spot += carry - 0.5 * variance * dt
        log_spot += np.sqrt(variance) * sqrt_dt * normal
        raw_force = (
            -parameters.feedback_rate * current_feedback
            + parameters.feedback_rate
            * parameters.feedback_volatility
            * np.sqrt(variance)
            * normal
            / sqrt_dt
        )
        raw_cell_increment = raw_force * weights[0] * dt
        balanced_force = raw_force / np.hypot(1.0, raw_cell_increment)
        forces[:, step] = balanced_force * dt
        feedback[:, step + 1] = parameters.initial_feedback + (
            forces[:, : step + 1] @ weights[: step + 1][::-1]
        )
    return SimulatedPaths(log_spot=log_spot, feedback=feedback)


def simulate_exponential_lift(
    parameters: QuadraticRoughHestonParameters,
    kernel: ExponentialKernel,
    maturity: float,
    normals: ArrayLike,
) -> SimulatedPaths:
    """Production-equivalent balanced recurrence written independently."""

    normal_values = np.atleast_2d(np.asarray(normals, dtype=np.float64))
    path_count, step_count = normal_values.shape
    nodes = np.asarray(kernel.nodes, dtype=np.float64)
    weights = np.asarray(kernel.weights, dtype=np.float64)
    dt = maturity / step_count
    sqrt_dt = math.sqrt(dt)
    decay = np.exp(-nodes * dt)
    drift_integral = np.where(
        nodes > 0.0, -np.expm1(-nodes * dt) / nodes, dt
    )
    feedback_cell_loading = float(weights @ drift_integral)
    factors = np.zeros((path_count, nodes.size), dtype=np.float64)
    feedback = np.empty((path_count, step_count + 1), dtype=np.float64)
    feedback[:, 0] = parameters.initial_feedback
    log_spot = np.full(path_count, math.log(parameters.spot), dtype=np.float64)
    carry = (parameters.risk_free_rate - parameters.dividend_yield) * dt

    for step in range(step_count):
        current_feedback = parameters.initial_feedback + factors @ weights
        variance = parameters.variance(current_feedback)
        normal = normal_values[:, step]
        log_spot += carry - 0.5 * variance * dt
        log_spot += np.sqrt(variance) * sqrt_dt * normal
        raw_force = (
            -parameters.feedback_rate * current_feedback
            + parameters.feedback_rate
            * parameters.feedback_volatility
            * np.sqrt(variance)
            * normal
            / sqrt_dt
        )
        raw_cell_increment = feedback_cell_loading * raw_force
        forcing = raw_force / np.hypot(1.0, raw_cell_increment)
        factors = decay[None, :] * factors
        factors += forcing[:, None] * drift_integral[None, :]
        feedback[:, step + 1] = parameters.initial_feedback + factors @ weights
    return SimulatedPaths(log_spot=log_spot, feedback=feedback)


__all__ = (
    "QuadraticRoughHestonParameters",
    "REFERENCE_PAPER",
    "SimulatedPaths",
    "UPSTREAM_REPOSITORY",
    "exponential_cell_average_weights",
    "fractional_cell_average_weights",
    "simulate_dense_convolution",
    "simulate_exponential_lift",
)
