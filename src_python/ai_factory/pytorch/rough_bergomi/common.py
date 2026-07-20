"""Rough Bergomi hybrid-scheme simulation helpers for lookback payoffs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from ai_factory.pytorch.common.device import seeded_generator


def parse_alpha(model: Mapping[str, Any]) -> float:
    if "alpha" in model:
        return float(model["alpha"])
    return float(model["hurst"]) - 0.5


def optimal_hybrid_evaluation_point(alpha: float, k: int) -> float:
    kd = float(k)
    average = (kd ** (alpha + 1.0) - (kd - 1.0) ** (alpha + 1.0)) / (
        alpha + 1.0
    )
    return average ** (1.0 / alpha)


def generate_max_spots(
    model: Mapping[str, Any],
    product: Mapping[str, Any],
    simulation: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate path maxima under rough Bergomi using the hybrid scheme."""

    target = torch.device(device)
    num_paths = int(simulation["num_paths"])
    num_steps = int(simulation["num_steps"])
    maturity = float(product["maturity"])
    dt = maturity / float(num_steps)
    sqrt_dt = math.sqrt(dt)

    spot0 = float(model["spot"])
    rate = float(model["risk_free_rate"])
    dividend_yield = float(model["dividend_yield"])
    forward_variance = float(model["forward_variance"])
    eta = float(model.get("eta", model.get("vol_of_vol")))
    alpha = parse_alpha(model)
    rho = float(model.get("rho", model.get("correlation")))

    if not (-0.5 < alpha < 0.0):
        raise ValueError("Rough Bergomi alpha must be in (-0.5, 0).")
    if forward_variance <= 0.0 or eta <= 0.0:
        raise ValueError("Rough Bergomi forward variance and eta must be positive.")
    if not (-1.0 < rho < 1.0):
        raise ValueError("Rough Bergomi rho must be in (-1, 1).")

    generator = seeded_generator(int(simulation["seed"]), target)
    w_normals = torch.randn(
        (num_paths, num_steps),
        generator=generator,
        device=target,
        dtype=dtype,
    )
    singular_normals = torch.randn(
        (num_paths, num_steps),
        generator=generator,
        device=target,
        dtype=dtype,
    )
    perpendicular_normals = torch.randn(
        (num_paths, num_steps),
        generator=generator,
        device=target,
        dtype=dtype,
    )

    dws = sqrt_dt * w_normals
    singular_covariance_scale = dt ** (alpha + 0.5) / (alpha + 1.0)
    singular_residual_variance = dt ** (2.0 * alpha + 1.0) * (
        1.0 / (2.0 * alpha + 1.0) - 1.0 / ((alpha + 1.0) * (alpha + 1.0))
    )
    singular_terms = (
        singular_covariance_scale * w_normals
        + math.sqrt(max(singular_residual_variance, 0.0)) * singular_normals
    )
    weights = torch.zeros(num_steps + 1, device=target, dtype=dtype)
    for k in range(2, num_steps + 1):
        weights[k] = (optimal_hybrid_evaluation_point(alpha, k) * dt) ** alpha

    log_spots = torch.full(
        (num_paths,),
        math.log(spot0),
        device=target,
        dtype=dtype,
    )
    max_spots = torch.full((num_paths,), spot0, device=target, dtype=dtype)
    drift = (rate - dividend_yield) * dt
    rho_perp = math.sqrt(1.0 - rho * rho)
    variance_scale = math.sqrt(2.0 * alpha + 1.0)

    for step in range(num_steps):
        y = torch.zeros((num_paths,), device=target, dtype=dtype)
        if step > 0:
            y = y + singular_terms[:, step - 1]
            for k in range(2, step + 1):
                y = y + weights[k] * dws[:, step - k]
        time = float(step) * dt
        variance = forward_variance * torch.exp(
            eta * variance_scale * y
            - 0.5 * eta * eta * (time ** (2.0 * alpha + 1.0))
        )
        dz = rho * dws[:, step] + rho_perp * sqrt_dt * perpendicular_normals[:, step]
        log_spots = log_spots + drift - 0.5 * variance * dt + torch.sqrt(variance) * dz
        max_spots = torch.maximum(max_spots, torch.exp(log_spots))

    return max_spots


def _tensor(values: list[float], target: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=target, dtype=dtype)


def generate_terminal_extrema_batch(
    models: list[Mapping[str, Any]],
    products: list[Mapping[str, Any]],
    simulations: list[Mapping[str, Any]],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate terminal, minimum, and maximum spots for a row batch.

    The implementation is vectorized over rows and Monte Carlo paths and uses
    the same code on CPU and CUDA. Python validation engines intentionally use
    native PyTorch RNG batches rather than row-wise reproducible streams.
    """

    if not (len(models) == len(products) == len(simulations)):
        raise ValueError("models, products, and simulations must have the same length.")
    if not models:
        return torch.empty((0, num_paths), device=torch.device(device), dtype=dtype)

    target = torch.device(device)
    batch_size = len(models)
    maturities = _tensor([float(product["maturity"]) for product in products], target, dtype)
    dt = maturities / float(num_steps)
    sqrt_dt = torch.sqrt(dt)

    spot0 = _tensor([float(model["spot"]) for model in models], target, dtype)
    rate = _tensor([float(model["risk_free_rate"]) for model in models], target, dtype)
    dividend_yield = _tensor(
        [float(model["dividend_yield"]) for model in models],
        target,
        dtype,
    )
    forward_variance = _tensor(
        [float(model["forward_variance"]) for model in models],
        target,
        dtype,
    )
    eta = _tensor(
        [float(model.get("eta", model.get("vol_of_vol"))) for model in models],
        target,
        dtype,
    )
    alpha = _tensor([parse_alpha(model) for model in models], target, dtype)
    rho = _tensor(
        [float(model.get("rho", model.get("correlation"))) for model in models],
        target,
        dtype,
    )

    if bool(torch.any((alpha <= -0.5) | (alpha >= 0.0)).cpu()):
        raise ValueError("Rough Bergomi alpha must be in (-0.5, 0).")
    if bool(torch.any((forward_variance <= 0.0) | (eta <= 0.0)).cpu()):
        raise ValueError("Rough Bergomi forward variance and eta must be positive.")
    if bool(torch.any((rho <= -1.0) | (rho >= 1.0)).cpu()):
        raise ValueError("Rough Bergomi rho must be in (-1, 1).")

    normal_shape = (batch_size, num_paths, num_steps)
    w_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    singular_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    perpendicular_normals = torch.randn(normal_shape, device=target, dtype=dtype)

    sqrt_dt_view = sqrt_dt[:, None, None]
    dws = sqrt_dt_view * w_normals
    singular_covariance_scale = dt ** (alpha + 0.5) / (alpha + 1.0)
    singular_residual_variance = dt ** (2.0 * alpha + 1.0) * (
        1.0 / (2.0 * alpha + 1.0) - 1.0 / ((alpha + 1.0) * (alpha + 1.0))
    )
    singular_terms = (
        singular_covariance_scale[:, None, None] * w_normals
        + torch.sqrt(torch.clamp(singular_residual_variance, min=0.0))[:, None, None]
        * singular_normals
    )

    lags = torch.arange(2, num_steps + 1, device=target, dtype=dtype)
    alpha_column = alpha[:, None]
    averages = (
        lags[None, :] ** (alpha_column + 1.0)
        - (lags[None, :] - 1.0) ** (alpha_column + 1.0)
    ) / (alpha_column + 1.0)
    positive_weights = (averages ** (1.0 / alpha_column) * dt[:, None]) ** alpha_column
    weights = torch.cat(
        (
            torch.zeros((batch_size, 2), device=target, dtype=dtype),
            positive_weights,
        ),
        dim=1,
    )

    # The hybrid history term is a causal convolution. Expressing it as one
    # batched matrix product avoids O(num_steps**2) Python-launched kernels.
    indices = torch.arange(num_steps, device=target)
    lag_matrix = indices[None, :] - indices[:, None]
    valid_lags = lag_matrix >= 2
    convolution_matrix = weights[:, lag_matrix.clamp(min=0)]
    convolution_matrix = convolution_matrix * valid_lags.to(dtype)
    history = torch.bmm(dws, convolution_matrix)
    singular_history = torch.nn.functional.pad(
        singular_terms[:, :, :-1],
        (1, 0),
    )
    y = history + singular_history

    times = torch.arange(num_steps, device=target, dtype=dtype)[None, :] * dt[:, None]
    variance = forward_variance[:, None, None] * torch.exp(
        eta[:, None, None] * torch.sqrt(2.0 * alpha + 1.0)[:, None, None] * y
        - 0.5
        * eta[:, None, None].square()
        * times[:, None, :] ** (2.0 * alpha[:, None, None] + 1.0)
    )
    dz = (
        rho[:, None, None] * dws
        + torch.sqrt(1.0 - rho.square())[:, None, None]
        * sqrt_dt[:, None, None]
        * perpendicular_normals
    )
    increments = (
        ((rate - dividend_yield) * dt)[:, None, None]
        - 0.5 * variance * dt[:, None, None]
        + torch.sqrt(variance) * dz
    )
    log_spots = torch.log(spot0)[:, None, None] + torch.cumsum(increments, dim=2)
    paths = torch.exp(log_spots)
    terminal = paths[:, :, -1]
    minimum = torch.minimum(paths.amin(dim=2), spot0[:, None])
    maximum = torch.maximum(paths.amax(dim=2), spot0[:, None])
    return torch.stack((terminal, minimum, maximum), dim=-1)


def generate_max_spots_batch(
    models: list[Mapping[str, Any]],
    products: list[Mapping[str, Any]],
    simulations: list[Mapping[str, Any]],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    state = generate_terminal_extrema_batch(
        models,
        products,
        simulations,
        num_paths=num_paths,
        num_steps=num_steps,
        device=device,
        dtype=dtype,
    )
    return state[:, :, 2]


def generate_average_spots_batch(
    models: list[Mapping[str, Any]],
    products: list[Mapping[str, Any]],
    simulations: list[Mapping[str, Any]],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate arithmetic path averages for a batch of Rough Bergomi rows.

    This mirrors :func:`generate_max_spots_batch`: it is vectorized over rows
    and Monte Carlo paths and only loops in Python over neither rows nor paths.
    """

    if not (len(models) == len(products) == len(simulations)):
        raise ValueError("models, products, and simulations must have the same length.")
    if not models:
        return torch.empty((0, num_paths), device=torch.device(device), dtype=dtype)

    target = torch.device(device)
    batch_size = len(models)
    maturities = _tensor([float(product["maturity"]) for product in products], target, dtype)
    dt = maturities / float(num_steps)
    sqrt_dt = torch.sqrt(dt)

    spot0 = _tensor([float(model["spot"]) for model in models], target, dtype)
    rate = _tensor([float(model["risk_free_rate"]) for model in models], target, dtype)
    dividend_yield = _tensor(
        [float(model["dividend_yield"]) for model in models],
        target,
        dtype,
    )
    forward_variance = _tensor(
        [float(model["forward_variance"]) for model in models],
        target,
        dtype,
    )
    eta = _tensor(
        [float(model.get("eta", model.get("vol_of_vol"))) for model in models],
        target,
        dtype,
    )
    alpha = _tensor([parse_alpha(model) for model in models], target, dtype)
    rho = _tensor(
        [float(model.get("rho", model.get("correlation"))) for model in models],
        target,
        dtype,
    )

    if bool(torch.any((alpha <= -0.5) | (alpha >= 0.0)).cpu()):
        raise ValueError("Rough Bergomi alpha must be in (-0.5, 0).")
    if bool(torch.any((forward_variance <= 0.0) | (eta <= 0.0)).cpu()):
        raise ValueError("Rough Bergomi forward variance and eta must be positive.")
    if bool(torch.any((rho <= -1.0) | (rho >= 1.0)).cpu()):
        raise ValueError("Rough Bergomi rho must be in (-1, 1).")

    normal_shape = (batch_size, num_paths, num_steps)
    w_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    singular_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    perpendicular_normals = torch.randn(normal_shape, device=target, dtype=dtype)

    dws = sqrt_dt[:, None, None] * w_normals
    singular_covariance_scale = dt ** (alpha + 0.5) / (alpha + 1.0)
    singular_residual_variance = dt ** (2.0 * alpha + 1.0) * (
        1.0 / (2.0 * alpha + 1.0) - 1.0 / ((alpha + 1.0) * (alpha + 1.0))
    )
    singular_terms = (
        singular_covariance_scale[:, None, None] * w_normals
        + torch.sqrt(torch.clamp(singular_residual_variance, min=0.0))[:, None, None]
        * singular_normals
    )

    lags = torch.arange(2, num_steps + 1, device=target, dtype=dtype)
    alpha_column = alpha[:, None]
    averages = (
        lags[None, :] ** (alpha_column + 1.0)
        - (lags[None, :] - 1.0) ** (alpha_column + 1.0)
    ) / (alpha_column + 1.0)
    positive_weights = (averages ** (1.0 / alpha_column) * dt[:, None]) ** alpha_column
    weights = torch.cat(
        (
            torch.zeros((batch_size, 2), device=target, dtype=dtype),
            positive_weights,
        ),
        dim=1,
    )

    indices = torch.arange(num_steps, device=target)
    lag_matrix = indices[None, :] - indices[:, None]
    valid_lags = lag_matrix >= 2
    convolution_matrix = weights[:, lag_matrix.clamp(min=0)]
    convolution_matrix = convolution_matrix * valid_lags.to(dtype)
    history = torch.bmm(dws, convolution_matrix)
    singular_history = torch.nn.functional.pad(
        singular_terms[:, :, :-1],
        (1, 0),
    )
    y = history + singular_history

    times = torch.arange(num_steps, device=target, dtype=dtype)[None, :] * dt[:, None]
    variance = forward_variance[:, None, None] * torch.exp(
        eta[:, None, None] * torch.sqrt(2.0 * alpha + 1.0)[:, None, None] * y
        - 0.5
        * eta[:, None, None].square()
        * times[:, None, :] ** (2.0 * alpha[:, None, None] + 1.0)
    )
    dz = (
        rho[:, None, None] * dws
        + torch.sqrt(1.0 - rho.square())[:, None, None]
        * sqrt_dt[:, None, None]
        * perpendicular_normals
    )
    increments = (
        ((rate - dividend_yield) * dt)[:, None, None]
        - 0.5 * variance * dt[:, None, None]
        + torch.sqrt(variance) * dz
    )
    log_spots = torch.log(spot0)[:, None, None] + torch.cumsum(increments, dim=2)
    return torch.exp(log_spots).mean(dim=2)


def generate_observation_spots_batch(
    models: list[Mapping[str, Any]],
    products: list[Mapping[str, Any]],
    simulations: list[Mapping[str, Any]],
    *,
    num_paths: int,
    num_steps: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate only the contractual observation spots for autocalls."""

    if not (len(models) == len(products) == len(simulations)):
        raise ValueError("models, products, and simulations must have the same length.")
    if not models:
        return torch.empty((0, num_paths, 0), device=torch.device(device), dtype=dtype)
    observation_counts = {int(product["observation_count"]) for product in products}
    if len(observation_counts) != 1:
        raise ValueError("A PyTorch autocall batch must share one observation count.")
    observation_count = observation_counts.pop()
    if num_steps % observation_count != 0:
        raise ValueError("Observation count must divide num_steps.")
    stride = num_steps // observation_count

    target = torch.device(device)
    batch_size = len(models)
    maturities = _tensor([float(product["maturity"]) for product in products], target, dtype)
    dt = maturities / float(num_steps)
    sqrt_dt = torch.sqrt(dt)
    spot0 = _tensor([float(model["spot"]) for model in models], target, dtype)
    rate = _tensor([float(model["risk_free_rate"]) for model in models], target, dtype)
    dividend_yield = _tensor(
        [float(model["dividend_yield"]) for model in models], target, dtype
    )
    forward_variance = _tensor(
        [float(model["forward_variance"]) for model in models], target, dtype
    )
    eta = _tensor(
        [float(model.get("eta", model.get("vol_of_vol"))) for model in models],
        target,
        dtype,
    )
    alpha = _tensor([parse_alpha(model) for model in models], target, dtype)
    rho = _tensor(
        [float(model.get("rho", model.get("correlation"))) for model in models],
        target,
        dtype,
    )

    normal_shape = (batch_size, num_paths, num_steps)
    w_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    singular_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    perpendicular_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    dws = sqrt_dt[:, None, None] * w_normals
    singular_covariance_scale = dt ** (alpha + 0.5) / (alpha + 1.0)
    singular_residual_variance = dt ** (2.0 * alpha + 1.0) * (
        1.0 / (2.0 * alpha + 1.0)
        - 1.0 / ((alpha + 1.0) * (alpha + 1.0))
    )
    singular_terms = (
        singular_covariance_scale[:, None, None] * w_normals
        + torch.sqrt(torch.clamp(singular_residual_variance, min=0.0))[:, None, None]
        * singular_normals
    )
    lags = torch.arange(2, num_steps + 1, device=target, dtype=dtype)
    alpha_column = alpha[:, None]
    averages = (
        lags[None, :] ** (alpha_column + 1.0)
        - (lags[None, :] - 1.0) ** (alpha_column + 1.0)
    ) / (alpha_column + 1.0)
    positive_weights = (averages ** (1.0 / alpha_column) * dt[:, None]) ** alpha_column
    weights = torch.cat(
        (torch.zeros((batch_size, 2), device=target, dtype=dtype), positive_weights),
        dim=1,
    )
    indices = torch.arange(num_steps, device=target)
    lag_matrix = indices[None, :] - indices[:, None]
    valid_lags = lag_matrix >= 2
    convolution_matrix = weights[:, lag_matrix.clamp(min=0)]
    convolution_matrix = convolution_matrix * valid_lags.to(dtype)
    history = torch.bmm(dws, convolution_matrix)
    singular_history = torch.nn.functional.pad(singular_terms[:, :, :-1], (1, 0))
    y = history + singular_history
    times = torch.arange(num_steps, device=target, dtype=dtype)[None, :] * dt[:, None]
    variance = forward_variance[:, None, None] * torch.exp(
        eta[:, None, None] * torch.sqrt(2.0 * alpha + 1.0)[:, None, None] * y
        - 0.5
        * eta[:, None, None].square()
        * times[:, None, :] ** (2.0 * alpha[:, None, None] + 1.0)
    )
    dz = (
        rho[:, None, None] * dws
        + torch.sqrt(1.0 - rho.square())[:, None, None]
        * sqrt_dt[:, None, None]
        * perpendicular_normals
    )
    increments = (
        ((rate - dividend_yield) * dt)[:, None, None]
        - 0.5 * variance * dt[:, None, None]
        + torch.sqrt(variance) * dz
    )
    log_spots = torch.log(spot0)[:, None, None] + torch.cumsum(increments, dim=2)
    return torch.exp(log_spots[:, :, stride - 1 :: stride])


def generate_realized_volatilities_batch(
    models: list[Mapping[str, Any]],
    products: list[Mapping[str, Any]],
    simulations: list[Mapping[str, Any]],
    *,
    num_paths: int,
    num_steps: int,
    observations_per_year: float = 52.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate annualized realized volatilities for Rough Bergomi rows.

    The weekly volatility swap convention uses ``sqrt(52 / N * sum r_i^2)``.
    The implementation mirrors the other rough Bergomi batch helpers: fully
    vectorized over rows and paths, with only PyTorch kernels doing the heavy
    lifting on either CPU or CUDA.
    """

    if not (len(models) == len(products) == len(simulations)):
        raise ValueError("models, products, and simulations must have the same length.")
    if not models:
        return torch.empty((0, num_paths), device=torch.device(device), dtype=dtype)

    target = torch.device(device)
    batch_size = len(models)
    maturities = _tensor([float(product["maturity"]) for product in products], target, dtype)
    dt = maturities / float(num_steps)
    sqrt_dt = torch.sqrt(dt)

    rate = _tensor([float(model["risk_free_rate"]) for model in models], target, dtype)
    dividend_yield = _tensor(
        [float(model["dividend_yield"]) for model in models],
        target,
        dtype,
    )
    forward_variance = _tensor(
        [float(model["forward_variance"]) for model in models],
        target,
        dtype,
    )
    eta = _tensor(
        [float(model.get("eta", model.get("vol_of_vol"))) for model in models],
        target,
        dtype,
    )
    alpha = _tensor([parse_alpha(model) for model in models], target, dtype)
    rho = _tensor(
        [float(model.get("rho", model.get("correlation"))) for model in models],
        target,
        dtype,
    )

    if bool(torch.any((alpha <= -0.5) | (alpha >= 0.0)).cpu()):
        raise ValueError("Rough Bergomi alpha must be in (-0.5, 0).")
    if bool(torch.any((forward_variance <= 0.0) | (eta <= 0.0)).cpu()):
        raise ValueError("Rough Bergomi forward variance and eta must be positive.")
    if bool(torch.any((rho <= -1.0) | (rho >= 1.0)).cpu()):
        raise ValueError("Rough Bergomi rho must be in (-1, 1).")

    normal_shape = (batch_size, num_paths, num_steps)
    w_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    singular_normals = torch.randn(normal_shape, device=target, dtype=dtype)
    perpendicular_normals = torch.randn(normal_shape, device=target, dtype=dtype)

    dws = sqrt_dt[:, None, None] * w_normals
    singular_covariance_scale = dt ** (alpha + 0.5) / (alpha + 1.0)
    singular_residual_variance = dt ** (2.0 * alpha + 1.0) * (
        1.0 / (2.0 * alpha + 1.0) - 1.0 / ((alpha + 1.0) * (alpha + 1.0))
    )
    singular_terms = (
        singular_covariance_scale[:, None, None] * w_normals
        + torch.sqrt(torch.clamp(singular_residual_variance, min=0.0))[:, None, None]
        * singular_normals
    )

    lags = torch.arange(2, num_steps + 1, device=target, dtype=dtype)
    alpha_column = alpha[:, None]
    averages = (
        lags[None, :] ** (alpha_column + 1.0)
        - (lags[None, :] - 1.0) ** (alpha_column + 1.0)
    ) / (alpha_column + 1.0)
    positive_weights = (averages ** (1.0 / alpha_column) * dt[:, None]) ** alpha_column
    weights = torch.cat(
        (
            torch.zeros((batch_size, 2), device=target, dtype=dtype),
            positive_weights,
        ),
        dim=1,
    )

    indices = torch.arange(num_steps, device=target)
    lag_matrix = indices[None, :] - indices[:, None]
    valid_lags = lag_matrix >= 2
    convolution_matrix = weights[:, lag_matrix.clamp(min=0)]
    convolution_matrix = convolution_matrix * valid_lags.to(dtype)
    history = torch.bmm(dws, convolution_matrix)
    singular_history = torch.nn.functional.pad(
        singular_terms[:, :, :-1],
        (1, 0),
    )
    y = history + singular_history

    times = torch.arange(num_steps, device=target, dtype=dtype)[None, :] * dt[:, None]
    variance = forward_variance[:, None, None] * torch.exp(
        eta[:, None, None] * torch.sqrt(2.0 * alpha + 1.0)[:, None, None] * y
        - 0.5
        * eta[:, None, None].square()
        * times[:, None, :] ** (2.0 * alpha[:, None, None] + 1.0)
    )
    dz = (
        rho[:, None, None] * dws
        + torch.sqrt(1.0 - rho.square())[:, None, None]
        * sqrt_dt[:, None, None]
        * perpendicular_normals
    )
    log_returns = (
        ((rate - dividend_yield) * dt)[:, None, None]
        - 0.5 * variance * dt[:, None, None]
        + torch.sqrt(variance) * dz
    )
    return torch.sqrt(
        (float(observations_per_year) / float(num_steps))
        * log_returns.square().sum(dim=2)
    )
