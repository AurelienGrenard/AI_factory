"""Batched Heston QE-M pathwise statistics for PyTorch products."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ai_factory.pytorch.common.pathwise_products import parameter_tensor
from ai_factory.pytorch.heston.common import QE_GAMMA_1, QE_GAMMA_2, QE_PSI_CRITICAL

PathStatistic = Literal[
    "terminal_spot", "max_spot", "average_spot", "realized_volatility", "observation_spots",
    "barrier_up", "barrier_down"
]


def simulate_statistic_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
    statistic: PathStatistic,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate one pathwise statistic per row/path under Heston QE-M."""

    target = torch.device(device)
    row_count = len(rows)
    normals = torch.randn(
        (row_count, num_paths, num_steps, 2),
        device=target,
        dtype=dtype,
    )
    uniforms = torch.rand(
        (row_count, num_paths, num_steps),
        device=target,
        dtype=dtype,
    )
    spot_shocks = normals[:, :, :, 0]
    stock_shocks = normals[:, :, :, 1]

    spot = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=target, dtype=dtype
    )
    rate = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "risk_free_rate",
        source="model",
        device=target,
        dtype=dtype,
    )
    dividend_yield = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "dividend_yield",
        source="model",
        device=target,
        dtype=dtype,
    )
    initial_variance = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "initial_variance",
        source="model",
        device=target,
        dtype=dtype,
    )
    kappa = parameter_tensor(
        rows, model_by_id, product_by_id, "kappa", source="model", device=target, dtype=dtype
    )
    theta = parameter_tensor(
        rows, model_by_id, product_by_id, "theta", source="model", device=target, dtype=dtype
    )
    xi = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "volatility_of_variance",
        source="model",
        device=target,
        dtype=dtype,
    )
    rho = parameter_tensor(
        rows, model_by_id, product_by_id, "rho", source="model", device=target, dtype=dtype
    )
    maturity = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "maturity",
        source="product",
        device=target,
        dtype=dtype,
    )

    dt = maturity / float(num_steps)
    exp_kdt = torch.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_kdt
    kappa_rho_over_xi = kappa * rho / xi
    rho_over_xi = rho / xi
    k1 = QE_GAMMA_1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi
    k2 = QE_GAMMA_2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi
    k3 = QE_GAMMA_1 * dt * (1.0 - rho * rho)
    k4 = QE_GAMMA_2 * dt * (1.0 - rho * rho)
    martingale_a = k2 + 0.5 * k4
    drift_dt = (rate - dividend_yield) * dt
    k0 = drift_dt - rho * kappa * theta * dt / xi

    log_spots = torch.log(spot).expand(-1, num_paths).clone()
    variances = initial_variance.expand(-1, num_paths).clone()
    observation_count = 0
    observation_stride = 0
    if statistic == "observation_spots":
        observation_counts = {
            int(product_by_id[row["product_id"]]["observation_count"]) for row in rows
        }
        if len(observation_counts) != 1:
            raise ValueError("A PyTorch autocall batch must share one observation count.")
        observation_count = observation_counts.pop()
        if num_steps % observation_count != 0:
            raise ValueError("Observation count must divide num_steps.")
        observation_stride = num_steps // observation_count
        statistic_values = torch.empty(
            (row_count, num_paths, observation_count), device=target, dtype=dtype
        )
    elif statistic in {"max_spot", "barrier_up", "barrier_down"}:
        statistic_values = spot.expand(-1, num_paths).clone()
    else:
        statistic_values = torch.zeros((row_count, num_paths), device=target, dtype=dtype)

    tiny = torch.finfo(dtype).tiny
    eps = torch.finfo(dtype).eps
    critical = torch.tensor(QE_PSI_CRITICAL, device=target, dtype=dtype)
    zero = torch.zeros((), device=target, dtype=dtype)
    one = torch.ones((), device=target, dtype=dtype)

    for step in range(num_steps):
        previous_variance = torch.clamp(variances, min=0.0)
        variance_normal = spot_shocks[:, :, step]
        stock_normal = stock_shocks[:, :, step]
        variance_uniform = uniforms[:, :, step]

        m = theta + (previous_variance - theta) * exp_kdt
        s2 = (
            previous_variance * xi * xi * exp_kdt * one_minus_exp / kappa
            + theta * xi * xi * one_minus_exp * one_minus_exp / (2.0 * kappa)
        )
        valid_moment = (m > 0.0) & (s2 > 0.0)
        safe_m = torch.where(valid_moment, m, one)
        safe_s2 = torch.where(valid_moment, s2, zero)
        psi = safe_s2 / (safe_m * safe_m)
        quadratic_branch = valid_moment & (psi <= critical)

        safe_psi = torch.clamp(psi, min=tiny)
        inv_psi = 1.0 / safe_psi
        b2 = (
            2.0 * inv_psi
            - 1.0
            + torch.sqrt(2.0 * inv_psi)
            * torch.sqrt(torch.clamp(2.0 * inv_psi - 1.0, min=0.0))
        )
        b2 = torch.clamp(b2, min=0.0)
        b = torch.sqrt(b2)
        a = safe_m / (1.0 + b2)
        shifted = b + variance_normal
        quadratic_variance = a * shifted * shifted
        quadratic_denominator = 1.0 - 2.0 * martingale_a * a
        quadratic_valid = quadratic_denominator > 0.0
        safe_denominator = torch.clamp(quadratic_denominator, min=tiny)
        quadratic_log_moment = (
            martingale_a * b2 * a / safe_denominator
            - 0.5 * torch.log(safe_denominator)
        )

        p = torch.clamp((psi - 1.0) / (psi + 1.0), min=0.0, max=1.0)
        beta = (1.0 - p) / safe_m
        safe_uniform = torch.clamp(variance_uniform, max=1.0 - eps)
        exponential_variance = torch.where(
            safe_uniform <= p,
            zero,
            torch.log((1.0 - p) / (1.0 - safe_uniform))
            / torch.clamp(beta, min=tiny),
        )
        exponential_valid = martingale_a < beta
        exponential_moment = p + beta * (1.0 - p) / torch.clamp(
            beta - martingale_a,
            min=tiny,
        )
        exponential_valid = exponential_valid & (exponential_moment > 0.0)
        exponential_log_moment = torch.log(torch.clamp(exponential_moment, min=tiny))

        next_variance = torch.where(
            valid_moment,
            torch.where(quadratic_branch, quadratic_variance, exponential_variance),
            zero,
        )
        log_moment = torch.where(
            quadratic_branch,
            quadratic_log_moment,
            exponential_log_moment,
        )
        martingale_valid = torch.where(
            quadratic_branch,
            quadratic_valid,
            exponential_valid,
        )
        martingale_valid = valid_moment & martingale_valid

        variance_integral_proxy = torch.clamp(
            k3 * previous_variance + k4 * next_variance,
            min=0.0,
        )
        uncorrected_increment = (
            k0
            + k1 * previous_variance
            + k2 * next_variance
            + torch.sqrt(variance_integral_proxy) * stock_normal
        )
        corrected_increment = (
            drift_dt
            - log_moment
            - 0.5 * k3 * previous_variance
            + k2 * next_variance
            + torch.sqrt(variance_integral_proxy) * stock_normal
        )
        previous_log_spots = log_spots
        log_spots = log_spots + torch.where(
            martingale_valid,
            corrected_increment,
            uncorrected_increment,
        )
        spot_values = torch.exp(log_spots)
        if statistic == "observation_spots":
            if (step + 1) % observation_stride == 0:
                statistic_values[:, :, (step + 1) // observation_stride - 1] = spot_values
        elif statistic in {"max_spot", "barrier_up"}:
            statistic_values = torch.maximum(statistic_values, spot_values)
        elif statistic == "barrier_down":
            statistic_values = torch.minimum(statistic_values, spot_values)
        elif statistic == "average_spot":
            statistic_values = statistic_values + spot_values
        else:
            statistic_values = statistic_values + (log_spots - previous_log_spots).square()
        variances = next_variance

    if statistic == "terminal_spot":
        statistic_values = torch.exp(log_spots)
    elif statistic == "average_spot":
        statistic_values = statistic_values / float(num_steps)
    elif statistic == "realized_volatility":
        statistic_values = torch.sqrt((52.0 / float(num_steps)) * statistic_values)
    elif statistic in {"barrier_up", "barrier_down"}:
        statistic_values = torch.stack((torch.exp(log_spots), statistic_values), dim=-1)

    discount = torch.exp(-rate * maturity)
    return statistic_values, discount, spot


def simulate_barrier_state_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    state, _, _ = simulate_statistic_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        statistic="barrier_up" if up else "barrier_down",
    )
    return state[:, :, 0], state[:, :, 1]


def simulate_observation_spots_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    observations, _, _ = simulate_statistic_batch(
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
        statistic="observation_spots",
    )
    return observations
