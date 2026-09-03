"""Independent Fourier references for rough and lifted Heston models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson, solve_ivp
from scipy.special import gamma

from validation.volterra.common import (
    ComplexArray,
    lewis_european_option_price,
    validate_option_inputs,
)


@dataclass(frozen=True)
class RoughHestonParameters:
    """Parameters in the same mathematical convention as the CUDA model.

    ``variance_drift`` is the constant theta in ``theta - lambda * V``;
    it is not the long-run variance.
    """

    spot: float
    risk_free_rate: float
    dividend_yield: float
    initial_variance: float
    mean_reversion: float
    variance_drift: float
    volatility_of_variance: float
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
        values = (
            self.initial_variance,
            self.mean_reversion,
            self.variance_drift,
            self.volatility_of_variance,
            self.hurst_exponent,
            self.rho,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rough-Heston parameters must be finite.")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive.")
        if min(
            self.mean_reversion,
            self.variance_drift,
            self.volatility_of_variance,
        ) < 0.0:
            raise ValueError(
                "rough-Heston drift and volatility coefficients must be "
                "non-negative."
            )
        if not 0.0 < self.hurst_exponent <= 0.5:
            raise ValueError("hurst_exponent must lie in (0, 0.5].")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1].")


@dataclass(frozen=True)
class ExponentialKernel:
    """One externally supplied sum of positive exponentials."""

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    initial_factors: tuple[float, ...] | None = None

    def arrays(
        self, initial_variance: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (
            nodes.ndim != 1
            or nodes.size == 0
            or weights.shape != nodes.shape
            or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(weights))
            or np.any(nodes < 0.0)
            or np.any(weights <= 0.0)
            or np.any(np.diff(nodes) <= 0.0)
        ):
            raise ValueError(
                "kernel nodes must be finite, ordered and non-negative; "
                "weights must be finite and positive."
            )
        if self.initial_factors is None:
            if np.any(nodes <= 0.0):
                raise ValueError(
                    "Positive nodes are required to reconstruct the default "
                    "anchored initial factors."
                )
            factors = initial_variance / nodes
            factors /= float(np.dot(weights, 1.0 / nodes))
        else:
            factors = np.asarray(self.initial_factors, dtype=np.float64)
            if (
                factors.shape != nodes.shape
                or not np.all(np.isfinite(factors))
            ):
                raise ValueError("initial_factors must match the kernel.")
        reconstructed = float(np.dot(weights, factors))
        if not math.isclose(
            reconstructed,
            initial_variance,
            rel_tol=2.0e-12,
            abs_tol=2.0e-14,
        ):
            raise ValueError(
                "kernel initial factors do not reconstruct initial_variance."
            )
        return nodes, weights, factors

    @classmethod
    def from_json(cls, path: str | Path) -> "ExponentialKernel":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read kernel JSON '{path}': {error}") from error
        if not isinstance(document, dict):
            raise ValueError("Kernel JSON must contain an object.")
        allowed = {"nodes", "weights", "initial_factors"}
        if not {"nodes", "weights"}.issubset(document) or not set(document) <= allowed:
            raise ValueError("Kernel JSON has an invalid field set.")
        initial = document.get("initial_factors")
        return cls(
            nodes=tuple(float(value) for value in document["nodes"]),
            weights=tuple(float(value) for value in document["weights"]),
            initial_factors=(
                tuple(float(value) for value in initial)
                if initial is not None
                else None
            ),
        )


def _riccati_polynomial(
    parameters: RoughHestonParameters,
    exponents: ComplexArray,
    values: ComplexArray,
) -> ComplexArray:
    return (
        0.5 * (exponents * exponents - exponents)
        + (
            parameters.rho
            * parameters.volatility_of_variance
            * exponents
            - parameters.mean_reversion
        )
        * values
        + 0.5
        * parameters.volatility_of_variance**2
        * values
        * values
    )


def fractional_riccati_grid(
    parameters: RoughHestonParameters,
    exponents: ArrayLike,
    maturity: float,
    time_step_count: int,
) -> tuple[NDArray[np.float64], ComplexArray, ComplexArray]:
    """Solve the fractional Riccati equation with Adams PECE.

    This deliberately slow ``O(time_step_count**2)`` implementation is an
    offline oracle.  It shares no discretization code with the CUDA model.
    """

    parameters.validate()
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if time_step_count < 2:
        raise ValueError("time_step_count must be at least 2.")
    z = np.atleast_1d(np.asarray(exponents, dtype=np.complex128))
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError("exponents must be a finite one-dimensional array.")

    alpha = parameters.hurst_exponent + 0.5
    dt = maturity / time_step_count
    times = np.linspace(0.0, maturity, time_step_count + 1)
    values = np.zeros((time_step_count + 1, z.size), dtype=np.complex128)
    right_hand_sides = np.empty_like(values)
    right_hand_sides[0] = _riccati_polynomial(parameters, z, values[0])
    predictor_scale = dt**alpha / gamma(alpha + 1.0)
    corrector_scale = dt**alpha / gamma(alpha + 2.0)

    for step in range(time_step_count):
        lags = np.arange(step + 1, 0, -1, dtype=np.float64)
        predictor_weights = lags**alpha - (lags - 1.0) ** alpha
        predictor = predictor_scale * np.einsum(
            "t,tf->f",
            predictor_weights,
            right_hand_sides[: step + 1],
            optimize=True,
        )

        corrector_weights = np.empty(step + 1, dtype=np.float64)
        corrector_weights[0] = step ** (alpha + 1.0) - (
            step - alpha
        ) * (step + 1.0) ** alpha
        if step > 0:
            inner_lags = np.arange(step - 1, -1, -1, dtype=np.float64)
            corrector_weights[1:] = (
                (inner_lags + 2.0) ** (alpha + 1.0)
                + inner_lags ** (alpha + 1.0)
                - 2.0 * (inner_lags + 1.0) ** (alpha + 1.0)
            )
        values[step + 1] = corrector_scale * (
            _riccati_polynomial(parameters, z, predictor)
            + np.einsum(
                "t,tf->f",
                corrector_weights,
                right_hand_sides[: step + 1],
                optimize=True,
            )
        )
        right_hand_sides[step + 1] = _riccati_polynomial(
            parameters, z, values[step + 1]
        )
        if not np.all(np.isfinite(values[step + 1])):
            raise RuntimeError(
                "Fractional Riccati solver diverged; refine or restrict the "
                "Fourier domain."
            )
    return times, values, right_hand_sides


def rough_heston_log_forward_mgf(
    parameters: RoughHestonParameters,
    exponents: ArrayLike,
    maturity: float,
    time_step_count: int,
) -> ComplexArray:
    """Return log E[(S_T/F_T)**z] for the continuous rough-Heston model."""

    times, riccati, right_hand_sides = fractional_riccati_grid(
        parameters, exponents, maturity, time_step_count
    )
    return (
        parameters.initial_variance
        * simpson(right_hand_sides, x=times, axis=0)
        + parameters.variance_drift
        * simpson(riccati, x=times, axis=0)
    )


def lifted_heston_log_forward_mgf(
    parameters: RoughHestonParameters,
    kernel: ExponentialKernel,
    exponents: ArrayLike,
    maturity: float,
    relative_tolerance: float = 2.0e-11,
    absolute_tolerance: float = 2.0e-13,
) -> ComplexArray:
    """Solve the affine Riccati ODE of exactly the supplied lifted model."""

    parameters.validate()
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    nodes, weights, initial = kernel.arrays(parameters.initial_variance)
    z_values = np.atleast_1d(np.asarray(exponents, dtype=np.complex128))
    if z_values.ndim != 1 or not np.all(np.isfinite(z_values)):
        raise ValueError("exponents must be a finite one-dimensional array.")
    factor_count = nodes.size
    state_width = factor_count + 1

    def derivative(_time: float, flat_state: ComplexArray) -> ComplexArray:
        state = flat_state.reshape(z_values.size, state_width)
        factors = state[:, :factor_count]
        aggregate = np.sum(factors, axis=1)
        polynomial = _riccati_polynomial(
            parameters, z_values, aggregate
        )
        output = np.empty_like(state)
        output[:, :factor_count] = (
            -nodes[None, :] * factors
            + weights[None, :] * polynomial[:, None]
        )
        output[:, factor_count] = factors @ (
            nodes * initial + parameters.variance_drift
        )
        return output.ravel()

    solution = solve_ivp(
        derivative,
        (0.0, maturity),
        np.zeros(z_values.size * state_width, dtype=np.complex128),
        method="DOP853",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise RuntimeError("Lifted Riccati ODE failed: " + str(solution.message))
    terminal = solution.y[:, -1].reshape(z_values.size, state_width)
    return terminal[:, factor_count] + terminal[:, :factor_count] @ initial


def rough_heston_european_option_price(
    parameters: RoughHestonParameters,
    strike: float,
    maturity: float,
    option_side: str,
    time_step_count: int = 1024,
    integration_cutoff: float = 80.0,
    integration_point_count: int = 1601,
) -> float:
    """European price under continuous rough Heston via fractional Riccati."""

    return lewis_european_option_price(
        lambda exponents: rough_heston_log_forward_mgf(
            parameters, exponents, maturity, time_step_count
        ),
        parameters.spot,
        strike,
        maturity,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        option_side,
        integration_cutoff,
        integration_point_count,
    )


def lifted_heston_european_option_price(
    parameters: RoughHestonParameters,
    kernel: ExponentialKernel,
    strike: float,
    maturity: float,
    option_side: str,
    integration_cutoff: float = 80.0,
    integration_point_count: int = 1601,
) -> float:
    """European price in the exact affine lift, without Monte Carlo."""

    return lewis_european_option_price(
        lambda exponents: lifted_heston_log_forward_mgf(
            parameters, kernel, exponents, maturity
        ),
        parameters.spot,
        strike,
        maturity,
        parameters.risk_free_rate,
        parameters.dividend_yield,
        option_side,
        integration_cutoff,
        integration_point_count,
    )


def parameters_from_mapping(value: Mapping[str, Any]) -> RoughHestonParameters:
    return RoughHestonParameters(
        spot=float(value["spot"]),
        risk_free_rate=float(value["risk_free_rate"]),
        dividend_yield=float(value["dividend_yield"]),
        initial_variance=float(value["initial_variance"]),
        mean_reversion=float(value["mean_reversion"]),
        variance_drift=float(value["variance_drift"]),
        volatility_of_variance=float(value["volatility_of_variance"]),
        hurst_exponent=float(value["hurst_exponent"]),
        rho=float(value["rho"]),
    )


__all__ = (
    "ExponentialKernel",
    "RoughHestonParameters",
    "fractional_riccati_grid",
    "lifted_heston_european_option_price",
    "lifted_heston_log_forward_mgf",
    "parameters_from_mapping",
    "rough_heston_european_option_price",
    "rough_heston_log_forward_mgf",
)
