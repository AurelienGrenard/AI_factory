"""Heston American put pricing with Longstaff-Schwartz in PyTorch."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.american_options import price_puts_from_paths_batch
from ai_factory.pytorch.common.pathwise_products import parameter_tensor
from ai_factory.pytorch.heston.common import (
    QE_GAMMA_1,
    QE_GAMMA_2,
    QE_PSI_CRITICAL,
)

DEFAULT_BATCH_ROWS = 8


def simulate_spot_paths_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Simulate batched Heston QE-M spot and variance paths."""

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
        rows,
        model_by_id,
        product_by_id,
        "spot",
        source="model",
        device=target,
        dtype=dtype,
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
        rows,
        model_by_id,
        product_by_id,
        "kappa",
        source="model",
        device=target,
        dtype=dtype,
    )
    theta = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "theta",
        source="model",
        device=target,
        dtype=dtype,
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
        rows,
        model_by_id,
        product_by_id,
        "rho",
        source="model",
        device=target,
        dtype=dtype,
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
    paths = torch.empty(
        (row_count, num_paths, num_steps + 1),
        device=target,
        dtype=dtype,
    )
    paths[:, :, 0] = spot
    variance_paths = torch.empty_like(paths)
    variance_paths[:, :, 0] = initial_variance

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
        log_spots = log_spots + torch.where(
            martingale_valid,
            corrected_increment,
            uncorrected_increment,
        )
        spot_values = torch.exp(log_spots)
        paths[:, :, step + 1] = spot_values
        variances = next_variance
        variance_paths[:, :, step + 1] = variances

    strike = parameter_tensor(
        rows,
        model_by_id,
        product_by_id,
        "strike",
        source="product",
        device=target,
        dtype=dtype,
    )
    return paths, variance_paths, maturity, rate, strike, theta


def price_from_paths(
    paths: torch.Tensor,
    *,
    variance_paths: torch.Tensor,
    theta: float,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    """Price one Heston American put from its complete Markov state paths."""

    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("American put LSM requires paths with at least two dates.")
    strikes = torch.tensor(
        [[strike]],
        device=paths.device,
        dtype=paths.dtype,
    )
    maturities = torch.tensor([[maturity]], device=paths.device, dtype=paths.dtype)
    rates = torch.tensor([[rate]], device=paths.device, dtype=paths.dtype)
    return price_from_paths_batch(
        paths.unsqueeze(0),
        variance_paths=variance_paths.unsqueeze(0),
        thetas=torch.tensor([[theta]], device=paths.device, dtype=paths.dtype),
        strikes=strikes,
        maturities=maturities,
        rates=rates,
    )[0]


def price_from_paths_batch(
    paths: torch.Tensor,
    *,
    variance_paths: torch.Tensor,
    thetas: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    rates: torch.Tensor,
) -> list[dict[str, float]]:
    """Use the Heston Markov state ``(S_t, V_t)`` in the LSM regression."""

    scaled_variance = variance_paths / thetas[:, None, :]
    l1_spot = 1.0 - paths / strikes.unsqueeze(1)
    features = torch.stack(
        (
            scaled_variance,
            scaled_variance.square(),
            l1_spot * scaled_variance,
        ),
        dim=3,
    )
    return price_puts_from_paths_batch(
        paths,
        strikes=strikes,
        maturities=maturities,
        rates=rates,
        regression_features=features,
    )


def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Price registry row batches on CPU or GPU with PyTorch."""

    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = perf_counter()
    simulation_seconds = 0.0
    lsm_seconds = 0.0
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        sim_started = perf_counter()
        paths, variance_paths, maturities, rates, strikes, thetas = simulate_spot_paths_batch(
            chunk,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=target,
            dtype=dtype,
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        simulation_seconds += perf_counter() - sim_started
        lsm_started = perf_counter()
        outputs.extend(
            price_from_paths_batch(
                paths,
                variance_paths=variance_paths,
                thetas=thetas,
                strikes=strikes,
                maturities=maturities,
                rates=rates,
            )
        )
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        lsm_seconds += perf_counter() - lsm_started
    return outputs, {
        "wall_seconds": perf_counter() - started,
        "simulation_seconds": simulation_seconds,
        "lsm_seconds": lsm_seconds,
    }


def price(
    model_parameters: Mapping[str, float],
    product_parameters: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    outputs, _ = price_batch(
        [{"model_id": "model", "product_id": "product"}],
        {"model": dict(model_parameters)},
        {"product": dict(product_parameters)},
        num_paths=int(simulation["num_paths"]),
        num_steps=int(simulation["num_steps"]),
        device=device,
        dtype=dtype,
        batch_rows=1,
    )
    return outputs[0]
