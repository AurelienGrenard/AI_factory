"""Independent FP64 hybrid references for the rough-SABR path mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from validation.volterra.common import (
    MonteCarloEstimate,
    estimate_from_antithetic_pairs,
    validate_option_inputs,
)
from validation.volterra.rough_bergomi import (
    hybrid_normalized_driver_direct,
    hybrid_normalized_driver_fft,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoughSabrParameters:
    """Rough-SABR convention used by the CUDA model.

    ``eta`` is the log-*variance* volatility and ``xi_0`` is the initial
    log-return variance at ``S_0``.  Hence the dimensional CEV coefficient is
    ``alpha_t = sqrt(xi_0) S_0^(1-beta)
    exp(eta Y_t / 2 - eta^2 t^(2H) / 4)``.  This convention makes ``beta=1``
    exactly the rough-Bergomi path mapping.
    """

    spot: float
    risk_free_rate: float
    dividend_yield: float
    xi_0: float
    eta: float
    hurst_exponent: float
    rho: float
    beta: float

    def validate(self) -> None:
        validate_option_inputs(
            self.spot,
            self.spot,
            1.0,
            self.risk_free_rate,
            self.dividend_yield,
        )
        values = (
            self.xi_0,
            self.eta,
            self.hurst_exponent,
            self.rho,
            self.beta,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rough-SABR parameters must be finite.")
        if self.xi_0 <= 0.0 or self.eta < 0.0:
            raise ValueError("xi_0 must be positive and eta non-negative.")
        if not 0.0 < self.hurst_exponent < 0.5:
            raise ValueError("hurst_exponent must lie in (0, 0.5).")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1].")
        if not 0.5 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0.5, 1].")


def rough_sabr_path_payoffs(
    parameters: RoughSabrParameters,
    strike: float,
    maturity: float,
    option_side: str,
    rough_normals: ArrayLike,
    singular_normals: ArrayLike,
    spot_normals: ArrayLike,
    convolution: Literal["direct", "fft"] = "direct",
) -> FloatArray:
    """Run the production Lamperti-spot mapping on supplied FP64 noise."""

    parameters.validate()
    validate_option_inputs(
        parameters.spot,
        strike,
        maturity,
        parameters.risk_free_rate,
        parameters.dividend_yield,
    )
    if option_side not in {"call", "put"}:
        raise ValueError("option_side must be 'call' or 'put'.")
    rough = np.atleast_2d(np.asarray(rough_normals, dtype=np.float64))
    singular = np.atleast_2d(np.asarray(singular_normals, dtype=np.float64))
    spot_noise = np.atleast_2d(np.asarray(spot_normals, dtype=np.float64))
    if rough.shape != singular.shape or rough.shape != spot_noise.shape:
        raise ValueError("rough-SABR random arrays must have identical shapes.")
    if rough.shape[1] < 1:
        raise ValueError("rough-SABR paths require at least one time step.")
    if convolution not in {"direct", "fft"}:
        raise ValueError("convolution must be 'direct' or 'fft'.")

    step_count = rough.shape[1]
    dt = maturity / step_count
    driver_function = (
        hybrid_normalized_driver_direct
        if convolution == "direct"
        else hybrid_normalized_driver_fft
    )
    driver = driver_function(
        parameters.hurst_exponent,
        maturity,
        math.sqrt(dt) * rough,
        rough,
        singular,
    )
    log_spot = np.full(rough.shape[0], math.log(parameters.spot))
    dimensional_alpha_0 = math.sqrt(parameters.xi_0) * parameters.spot ** (
        1.0 - parameters.beta
    )
    volatility = np.full(rough.shape[0], dimensional_alpha_0)
    orthogonal_correlation = math.sqrt(max(1.0 - parameters.rho**2, 0.0))
    correlated = (
        parameters.rho * rough + orthogonal_correlation * spot_noise
    )
    carry_step = (parameters.risk_free_rate - parameters.dividend_yield) * dt
    sqrt_dt = math.sqrt(dt)
    for step in range(step_count):
        one_minus_beta = 1.0 - parameters.beta
        if one_minus_beta < 1.0e-4:
            log_spot += (
                carry_step
                - 0.5 * volatility**2 * dt
                + volatility * sqrt_dt * correlated[:, step]
            )
        else:
            spot = np.exp(log_spot)
            transformed = spot**one_minus_beta / one_minus_beta
            transformed = np.maximum(
                transformed
                + one_minus_beta * transformed * carry_step
                + volatility * sqrt_dt * correlated[:, step]
                - 0.5
                * parameters.beta
                * volatility**2
                * dt
                / (one_minus_beta * transformed),
                1.0e-12,
            )
            log_spot = np.log(one_minus_beta * transformed) / one_minus_beta
        time = (step + 1) * dt
        volatility = dimensional_alpha_0 * np.exp(
            0.5 * parameters.eta * driver[:, step]
            - 0.25
            * parameters.eta**2
            * time ** (2.0 * parameters.hurst_exponent)
        )
    terminal = np.exp(log_spot)
    intrinsic = (
        np.maximum(terminal - strike, 0.0)
        if option_side == "call"
        else np.maximum(strike - terminal, 0.0)
    )
    return math.exp(-parameters.risk_free_rate * maturity) * intrinsic


def hybrid_european_option_price(
    parameters: RoughSabrParameters,
    strike: float,
    maturity: float,
    option_side: str,
    time_step_count: int,
    antithetic_pair_count: int,
    seed: int,
    batch_pair_count: int = 4096,
    convolution: Literal["direct", "fft"] = "fft",
) -> MonteCarloEstimate:
    """Price rough SABR with an auditable batched NumPy hybrid scheme."""

    parameters.validate()
    if time_step_count < 1 or antithetic_pair_count < 2:
        raise ValueError("Positive steps and at least two pairs are required.")
    if batch_pair_count < 1:
        raise ValueError("batch_pair_count must be positive.")
    generator = np.random.default_rng(seed)
    positive_parts: list[FloatArray] = []
    negative_parts: list[FloatArray] = []
    remaining = antithetic_pair_count
    while remaining:
        batch = min(remaining, batch_pair_count)
        noise = generator.standard_normal(
            (3, batch, time_step_count), dtype=np.float64
        )
        positive_parts.append(
            rough_sabr_path_payoffs(
                parameters,
                strike,
                maturity,
                option_side,
                noise[0],
                noise[1],
                noise[2],
                convolution,
            )
        )
        negative_parts.append(
            rough_sabr_path_payoffs(
                parameters,
                strike,
                maturity,
                option_side,
                -noise[0],
                -noise[1],
                -noise[2],
                convolution,
            )
        )
        remaining -= batch
    return estimate_from_antithetic_pairs(
        np.concatenate(positive_parts), np.concatenate(negative_parts)
    )


__all__ = (
    "RoughSabrParameters",
    "hybrid_european_option_price",
    "rough_sabr_path_payoffs",
)
