"""Vectorized curve and one-factor Gaussian-rate formulas."""

from __future__ import annotations

import torch


def shifted_black_option(
    forward: torch.Tensor,
    strike: torch.Tensor,
    displacement: torch.Tensor,
    total_volatility: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    shifted_forward = forward + displacement
    shifted_strike = strike + displacement
    d1 = (
        torch.log(shifted_forward / shifted_strike) / total_volatility
        + 0.5 * total_volatility
    )
    d2 = d1 - total_volatility
    normal = torch.distributions.Normal(
        torch.zeros((), device=forward.device, dtype=forward.dtype),
        torch.ones((), device=forward.device, dtype=forward.dtype),
    )
    return direction * (
        shifted_forward * normal.cdf(direction * d1)
        - shifted_strike * normal.cdf(direction * d2)
    )


def nelson_siegel_zero_rate(
    maturity: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    x = maturity / tau
    loading = -torch.expm1(-x) / x
    return beta0 + beta1 * loading + beta2 * (loading - torch.exp(-x))


def nelson_siegel_discount(
    maturity: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(
        -maturity
        * nelson_siegel_zero_rate(maturity, beta0, beta1, beta2, tau)
    )


def hull_white_b(mean_reversion: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-mean_reversion * horizon) / mean_reversion


def hull_white_state_variance(
    mean_reversion: torch.Tensor,
    volatility: torch.Tensor,
    horizon: torch.Tensor,
) -> torch.Tensor:
    return volatility.square() * -torch.expm1(-2.0 * mean_reversion * horizon) / (
        2.0 * mean_reversion
    )


def hull_white_integral_variance(
    mean_reversion: torch.Tensor,
    volatility: torch.Tensor,
    horizon: torch.Tensor,
) -> torch.Tensor:
    one_minus_e = -torch.expm1(-mean_reversion * horizon)
    one_minus_e2 = -torch.expm1(-2.0 * mean_reversion * horizon)
    bracket = (
        horizon
        - 2.0 * one_minus_e / mean_reversion
        + one_minus_e2 / (2.0 * mean_reversion)
    )
    return volatility.square() * bracket / mean_reversion.square()


def hull_white_state_integral_covariance(
    mean_reversion: torch.Tensor,
    volatility: torch.Tensor,
    horizon: torch.Tensor,
) -> torch.Tensor:
    one_minus_e = -torch.expm1(-mean_reversion * horizon)
    return volatility.square() * one_minus_e.square() / (2.0 * mean_reversion.square())


def hull_white_bond_price(
    state: torch.Tensor,
    time: torch.Tensor,
    maturity: torch.Tensor,
    mean_reversion: torch.Tensor,
    volatility: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    p0_time = nelson_siegel_discount(time, beta0, beta1, beta2, tau)
    p0_maturity = nelson_siegel_discount(maturity, beta0, beta1, beta2, tau)
    adjustment = (
        hull_white_integral_variance(mean_reversion, volatility, maturity)
        - hull_white_integral_variance(mean_reversion, volatility, time)
        - hull_white_integral_variance(mean_reversion, volatility, maturity - time)
    )
    return (p0_maturity / p0_time) * torch.exp(
        -hull_white_b(mean_reversion, maturity - time) * state - 0.5 * adjustment
    )


def hull_white_bond_coefficients(
    time: torch.Tensor,
    maturity: torch.Tensor,
    mean_reversion: torch.Tensor,
    volatility: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p0_time = nelson_siegel_discount(time, beta0, beta1, beta2, tau)
    p0_maturity = nelson_siegel_discount(maturity, beta0, beta1, beta2, tau)
    adjustment = (
        hull_white_integral_variance(mean_reversion, volatility, maturity)
        - hull_white_integral_variance(mean_reversion, volatility, time)
        - hull_white_integral_variance(mean_reversion, volatility, maturity - time)
    )
    bond_a = (p0_maturity / p0_time) * torch.exp(-0.5 * adjustment)
    bond_b = hull_white_b(mean_reversion, maturity - time)
    return bond_a, bond_b


def cir_bond_price(
    short_rate: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    volatility: torch.Tensor,
    horizon: torch.Tensor,
) -> torch.Tensor:
    gamma = torch.sqrt(kappa.square() + 2.0 * volatility.square())
    exponential_minus_one = torch.expm1(gamma * horizon)
    denominator = (gamma + kappa) * exponential_minus_one + 2.0 * gamma
    bond_b = 2.0 * exponential_minus_one / denominator
    log_base = (
        torch.log(2.0 * gamma / denominator)
        + 0.5 * (kappa + gamma) * horizon
    )
    bond_a = torch.exp(
        2.0 * kappa * theta / volatility.square() * log_base
    )
    return bond_a * torch.exp(-bond_b * short_rate)


def cir_bond_coefficients(
    kappa: torch.Tensor,
    theta: torch.Tensor,
    volatility: torch.Tensor,
    horizon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.sqrt(kappa.square() + 2.0 * volatility.square())
    exponential_minus_one = torch.expm1(gamma * horizon)
    denominator = (gamma + kappa) * exponential_minus_one + 2.0 * gamma
    bond_b = 2.0 * exponential_minus_one / denominator
    log_base = (
        torch.log(2.0 * gamma / denominator)
        + 0.5 * (kappa + gamma) * horizon
    )
    bond_a = torch.exp(
        2.0 * kappa * theta / volatility.square() * log_base
    )
    return bond_a, bond_b


def cir_plus_plus_bond_coefficients(
    time: torch.Tensor,
    maturity: torch.Tensor,
    initial_factor: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    volatility: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_a_t, base_b_t = cir_bond_coefficients(kappa, theta, volatility, time)
    base_a_m, base_b_m = cir_bond_coefficients(
        kappa, theta, volatility, maturity
    )
    horizon_a, horizon_b = cir_bond_coefficients(
        kappa, theta, volatility, maturity - time
    )
    market_ratio = nelson_siegel_discount(
        maturity, beta0, beta1, beta2, tau
    ) / nelson_siegel_discount(time, beta0, beta1, beta2, tau)
    base_ratio = (
        base_a_t * torch.exp(-base_b_t * initial_factor)
        / (base_a_m * torch.exp(-base_b_m * initial_factor))
    )
    return market_ratio * base_ratio * horizon_a, horizon_b


def cir_plus_plus_path_discount(
    integrated_factor: torch.Tensor,
    time: torch.Tensor,
    initial_factor: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    volatility: torch.Tensor,
    beta0: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    base_discount = cir_bond_price(
        initial_factor, kappa, theta, volatility, time
    )
    return (
        nelson_siegel_discount(time, beta0, beta1, beta2, tau)
        / base_discount
        * torch.exp(-integrated_factor)
    )


def g2_integral_variance(a, sigma, b, eta, rho, horizon):
    cross = rho * sigma * eta / (a * b) * (
        horizon
        - (-torch.expm1(-a * horizon)) / a
        - (-torch.expm1(-b * horizon)) / b
        + (-torch.expm1(-(a + b) * horizon)) / (a + b)
    )
    return (
        hull_white_integral_variance(a, sigma, horizon)
        + hull_white_integral_variance(b, eta, horizon)
        + 2.0 * cross
    )


def g2_bond_coefficients(
    time, maturity, a, sigma, b, eta, rho, beta0, beta1, beta2, tau
):
    horizon = maturity - time
    adjustment = 0.5 * (
        g2_integral_variance(a, sigma, b, eta, rho, horizon)
        - g2_integral_variance(a, sigma, b, eta, rho, maturity)
        + g2_integral_variance(a, sigma, b, eta, rho, time)
    )
    bond_a = (
        nelson_siegel_discount(maturity, beta0, beta1, beta2, tau)
        / nelson_siegel_discount(time, beta0, beta1, beta2, tau)
        * torch.exp(adjustment)
    )
    return bond_a, hull_white_b(a, horizon), hull_white_b(b, horizon)


def g2_path_discount(
    integrated_x, integrated_y, time, a, sigma, b, eta, rho,
    beta0, beta1, beta2, tau,
):
    return nelson_siegel_discount(time, beta0, beta1, beta2, tau) * torch.exp(
        -0.5 * g2_integral_variance(a, sigma, b, eta, rho, time)
        - integrated_x
        - integrated_y
    )


def g2_transition(a, sigma, b, eta, rho, horizon):
    one_a = -torch.expm1(-a * horizon)
    one_b = -torch.expm1(-b * horizon)
    one_ab = -torch.expm1(-(a + b) * horizon)
    covariance = torch.zeros(
        (*a.shape[:-1], 4, 4), device=a.device, dtype=a.dtype
    )
    covariance[..., 0, 0] = hull_white_state_variance(a, sigma, horizon).squeeze(-1)
    covariance[..., 1, 1] = hull_white_state_variance(b, eta, horizon).squeeze(-1)
    covariance[..., 2, 2] = hull_white_integral_variance(a, sigma, horizon).squeeze(-1)
    covariance[..., 3, 3] = hull_white_integral_variance(b, eta, horizon).squeeze(-1)
    covariance[..., 0, 1] = covariance[..., 1, 0] = (
        rho * sigma * eta * one_ab / (a + b)
    ).squeeze(-1)
    covariance[..., 0, 2] = covariance[..., 2, 0] = (
        sigma.square() * one_a.square() / (2.0 * a.square())
    ).squeeze(-1)
    covariance[..., 1, 3] = covariance[..., 3, 1] = (
        eta.square() * one_b.square() / (2.0 * b.square())
    ).squeeze(-1)
    covariance[..., 0, 3] = covariance[..., 3, 0] = (
        rho * sigma * eta / b * (one_a / a - one_ab / (a + b))
    ).squeeze(-1)
    covariance[..., 1, 2] = covariance[..., 2, 1] = (
        rho * sigma * eta / a * (one_b / b - one_ab / (a + b))
    ).squeeze(-1)
    covariance[..., 2, 3] = covariance[..., 3, 2] = (
        rho * sigma * eta / (a * b)
        * (horizon - one_a / a - one_b / b + one_ab / (a + b))
    ).squeeze(-1)
    return (
        torch.exp(-a * horizon),
        torch.exp(-b * horizon),
        hull_white_b(a, horizon),
        hull_white_b(b, horizon),
        torch.linalg.cholesky(covariance),
    )
