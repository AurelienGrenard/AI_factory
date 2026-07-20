"""Heston path simulation with Euler and Andersen QE variance schemes."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import (
    resolve_device,
    resolve_random_device,
    seeded_generator,
    synchronize as _synchronize_if_needed,
)

QE_PSI_CRITICAL = 1.5
QE_GAMMA_1 = 0.5
QE_GAMMA_2 = 0.5


def _canonical_scheme(value: object) -> str:
    scheme = str(value or "euler")
    if scheme in {"euler", "euler_full_truncation"}:
        return "euler"
    if scheme in {"qe", "andersen_qe"}:
        return "qe"
    if scheme in {"qe_martingale", "andersen_qe_martingale", "qe-m"}:
        return "qe_martingale"
    raise ValueError(f"Unsupported Heston simulation scheme: {scheme}")


def generate_paths(
    model_parameters: Mapping[str, float],
    product_parameters: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate Heston spot paths.

    By default, random draws happen directly on the requested output device.
    Set simulation["random_device"] explicitly only when a different random
    device is needed for a specific reference check.
    """

    output_device = resolve_device(device)
    simulation_device = resolve_random_device(
        str(simulation.get("random_device", "target")),
        output_device,
    )
    seed = int(simulation["seed"])
    num_paths = int(simulation["num_paths"])
    num_steps = int(simulation["num_steps"])
    maturity = float(product_parameters["maturity"])

    spot = float(model_parameters["spot"])
    rate = float(model_parameters["risk_free_rate"])
    dividend_yield = float(model_parameters.get("dividend_yield", 0.0))
    initial_variance = float(model_parameters["initial_variance"])
    kappa = float(model_parameters["kappa"])
    theta = float(model_parameters["theta"])
    volatility_of_variance = float(model_parameters["volatility_of_variance"])
    rho = float(model_parameters["rho"])

    dt = maturity / num_steps
    sqrt_dt = dt**0.5
    scheme = _canonical_scheme(simulation.get("heston_scheme", "euler"))
    random_backend = str(simulation.get("random_backend", "pytorch_randn"))
    if random_backend not in {"pytorch_randn", "torch_randn"}:
        raise ValueError(f"Unsupported Python random_backend: {random_backend}")
    generator = seeded_generator(seed, simulation_device)
    independent_shocks = torch.randn(
        (num_paths, num_steps, 2),
        generator=generator,
        device=simulation_device,
        dtype=dtype,
    )
    spot_shocks = independent_shocks[:, :, 0]
    independent_variance_shocks = independent_shocks[:, :, 1]
    qe_uniforms = torch.rand(
        (num_paths, num_steps),
        generator=generator,
        device=simulation_device,
        dtype=dtype,
    )

    paths = torch.empty(
        (num_paths, num_steps + 1),
        device=simulation_device,
        dtype=dtype,
    )
    variances = torch.empty((num_paths,), device=simulation_device, dtype=dtype)
    paths[:, 0] = spot
    variances.fill_(initial_variance)

    if scheme == "euler":
        variance_shocks = rho * spot_shocks + (1.0 - rho**2) ** 0.5 * (
            independent_variance_shocks
        )
        for step in range(num_steps):
            variance_floor = torch.clamp(variances, min=0.0)
            paths[:, step + 1] = paths[:, step] * torch.exp(
                (rate - dividend_yield - 0.5 * variance_floor) * dt
                + torch.sqrt(variance_floor) * sqrt_dt * spot_shocks[:, step]
            )
            variances = variances + kappa * (theta - variance_floor) * dt
            variances = variances + volatility_of_variance * torch.sqrt(
                variance_floor
            ) * sqrt_dt * variance_shocks[:, step]
            variances = torch.clamp(variances, min=0.0)
    else:
        xi = volatility_of_variance
        if xi <= 0.0:
            raise ValueError("Heston QE schemes require positive volatility_of_variance.")
        rho_tensor = torch.as_tensor(rho, device=simulation_device, dtype=dtype)
        kappa_rho_over_xi = kappa * rho / xi
        rho_over_xi = rho / xi
        k1 = QE_GAMMA_1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi
        k2 = QE_GAMMA_2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi
        k3 = QE_GAMMA_1 * dt * (1.0 - rho * rho)
        k4 = QE_GAMMA_2 * dt * (1.0 - rho * rho)
        martingale_a = k2 + 0.5 * k4
        exp_kdt = torch.as_tensor(
            torch.exp(torch.as_tensor(-kappa * dt, device=simulation_device, dtype=dtype)),
            device=simulation_device,
            dtype=dtype,
        )
        one_minus_exp = 1.0 - exp_kdt
        log_spots = torch.full(
            (num_paths,),
            torch.log(torch.as_tensor(spot, device=simulation_device, dtype=dtype)),
            device=simulation_device,
            dtype=dtype,
        )
        drift_dt = (rate - dividend_yield) * dt
        k0 = drift_dt - rho * kappa * theta * dt / xi
        zero = torch.zeros((), device=simulation_device, dtype=dtype)
        one = torch.ones((), device=simulation_device, dtype=dtype)
        critical = torch.as_tensor(QE_PSI_CRITICAL, device=simulation_device, dtype=dtype)

        for step in range(num_steps):
            previous_variance = torch.clamp(variances, min=0.0)
            variance_normal = spot_shocks[:, step]
            stock_normal = independent_variance_shocks[:, step]
            variance_uniform = qe_uniforms[:, step]

            m = theta + (previous_variance - theta) * exp_kdt
            s2 = (
                previous_variance
                * xi
                * xi
                * exp_kdt
                * one_minus_exp
                / kappa
                + theta
                * xi
                * xi
                * one_minus_exp
                * one_minus_exp
                / (2.0 * kappa)
            )
            valid_moment = (m > 0.0) & (s2 > 0.0)
            safe_m = torch.where(valid_moment, m, one)
            safe_s2 = torch.where(valid_moment, s2, zero)
            psi = safe_s2 / (safe_m * safe_m)
            quadratic_branch = valid_moment & (psi <= critical)

            safe_psi = torch.clamp(psi, min=torch.finfo(dtype).tiny)
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
            quadratic_log_moment = (
                martingale_a * b2 * a / torch.clamp(quadratic_denominator, min=torch.finfo(dtype).tiny)
                - 0.5 * torch.log(torch.clamp(quadratic_denominator, min=torch.finfo(dtype).tiny))
            )

            p = torch.clamp((psi - 1.0) / (psi + 1.0), min=0.0, max=1.0)
            beta = (1.0 - p) / safe_m
            safe_uniform = torch.clamp(
                variance_uniform,
                max=1.0 - torch.finfo(dtype).eps,
            )
            exponential_variance = torch.where(
                safe_uniform <= p,
                zero,
                torch.log((1.0 - p) / (1.0 - safe_uniform))
                / torch.clamp(beta, min=torch.finfo(dtype).tiny),
            )
            exponential_valid = martingale_a < beta
            exponential_moment = p + beta * (1.0 - p) / torch.clamp(
                beta - martingale_a,
                min=torch.finfo(dtype).tiny,
            )
            exponential_valid = exponential_valid & (exponential_moment > 0.0)
            exponential_log_moment = torch.log(
                torch.clamp(exponential_moment, min=torch.finfo(dtype).tiny)
            )

            next_variance = torch.where(
                valid_moment,
                torch.where(
                    quadratic_branch,
                    quadratic_variance,
                    exponential_variance,
                ),
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
            if scheme == "qe_martingale":
                log_spots = log_spots + torch.where(
                    martingale_valid,
                    corrected_increment,
                    uncorrected_increment,
                )
            else:
                log_spots = log_spots + uncorrected_increment
            variances = next_variance
            paths[:, step + 1] = torch.exp(log_spots)

    return paths.to(output_device)


def generate_paths_with_timing(
    model_parameters: Mapping[str, float],
    product_parameters: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generate paths and return wall-clock timing metadata."""

    output_device = resolve_device(device)
    _synchronize_if_needed(output_device)
    start = perf_counter()
    paths = generate_paths(
        model_parameters,
        product_parameters,
        simulation,
        device=device,
        dtype=dtype,
    )
    _synchronize_if_needed(output_device)
    elapsed = perf_counter() - start
    return paths, {"simulation_seconds": elapsed}
