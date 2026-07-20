"""Batched multifactor rough Heston simulation for PyTorch products."""

from __future__ import annotations

from typing import Any, Literal

import torch

from ai_factory.pytorch.common.pathwise_products import parameter_tensor

FACTOR_COUNT = 8
PathStatistic = Literal[
    "terminal_spot", "max_spot", "average_spot", "realized_volatility",
    "observation_spots", "spot_paths", "state_paths", "barrier_up", "barrier_down",
]


def _factor_coefficients(
    hurst: torch.Tensor,
    maturity: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = hurst + 0.5
    measure_scale = torch.sin(torch.pi * alpha) / torch.pi
    lower = 0.1 / maturity
    upper = 20.0 / dt
    ratio = (upper / lower) ** (1.0 / float(FACTOR_COUNT - 1))
    indices = torch.arange(FACTOR_COUNT, device=hurst.device, dtype=hurst.dtype)
    right = lower * ratio ** indices
    left = torch.cat((torch.zeros_like(lower), right[:, :-1]), dim=1)
    mass = measure_scale * (
        right ** (1.0 - alpha) - left ** (1.0 - alpha)
    ) / (1.0 - alpha)
    first_moment = measure_scale * (
        right ** (2.0 - alpha) - left ** (2.0 - alpha)
    ) / (2.0 - alpha)
    nodes = first_moment / mass
    decay = torch.exp(-nodes * dt)
    drift = mass * (-torch.expm1(-nodes * dt)) / nodes
    return decay, drift, drift / torch.sqrt(dt)


def simulate_statistic_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    num_paths: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
    *,
    statistic: PathStatistic = "terminal_spot",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.device(device)
    row_count = len(rows)
    shape = (row_count, num_paths, num_steps)
    variance_normals = torch.randn(shape, device=target, dtype=dtype)
    independent_normals = torch.randn(shape, device=target, dtype=dtype)

    def parameter(name: str, source: str = "model") -> torch.Tensor:
        return parameter_tensor(
            rows, model_by_id, product_by_id, name,
            source=source, device=target, dtype=dtype,
        )

    spot0 = parameter("spot")
    rate = parameter("risk_free_rate")
    dividend = parameter("dividend_yield")
    initial_variance = parameter("initial_variance")
    kappa = parameter("kappa")
    theta = parameter("theta")
    volatility = parameter("volatility_of_variance")
    hurst = parameter("hurst")
    rho = parameter("rho")
    maturity = parameter("maturity", "product")
    dt = maturity / float(num_steps)
    decay, drift_weight, diffusion_weight = _factor_coefficients(
        hurst, maturity, dt
    )

    log_spot = torch.log(spot0).expand(-1, num_paths).clone()
    variance = initial_variance.expand(-1, num_paths).clone()
    factors = torch.zeros(
        (row_count, num_paths, FACTOR_COUNT), device=target, dtype=dtype
    )
    observation_count = 0
    observation_stride = 0
    factor_paths = None
    if statistic in {"spot_paths", "state_paths"}:
        values = torch.empty(
            (row_count, num_paths, num_steps + 1), device=target, dtype=dtype
        )
        values[:, :, 0] = spot0
        if statistic == "state_paths":
            factor_paths = torch.empty(
                (row_count, num_paths, num_steps + 1, FACTOR_COUNT),
                device=target,
                dtype=dtype,
            )
            factor_paths[:, :, 0, :] = 0.0
    elif statistic == "observation_spots":
        counts = {int(product_by_id[row["product_id"]]["observation_count"]) for row in rows}
        if len(counts) != 1:
            raise ValueError("A rough Heston autocall batch must share observation_count.")
        observation_count = counts.pop()
        if num_steps % observation_count:
            raise ValueError("Observation count must divide num_steps.")
        observation_stride = num_steps // observation_count
        values = torch.empty(
            (row_count, num_paths, observation_count), device=target, dtype=dtype
        )
    elif statistic in {"max_spot", "barrier_up", "barrier_down"}:
        values = spot0.expand(-1, num_paths).clone()
    else:
        values = torch.zeros((row_count, num_paths), device=target, dtype=dtype)

    rho_perp = torch.sqrt(1.0 - rho.square())
    sqrt_dt = torch.sqrt(dt)
    for step in range(num_steps):
        positive_variance = torch.clamp(variance, min=0.0)
        root_variance = torch.sqrt(positive_variance)
        z_variance = variance_normals[:, :, step]
        common_drift = kappa * (theta - positive_variance)
        common_diffusion = volatility * root_variance * z_variance
        factors = (
            decay[:, None, :] * factors
            + drift_weight[:, None, :] * common_drift[:, :, None]
            + diffusion_weight[:, None, :] * common_diffusion[:, :, None]
        )
        previous_log_spot = log_spot
        stock_normal = (
            rho * z_variance + rho_perp * independent_normals[:, :, step]
        )
        log_spot = log_spot + (
            (rate - dividend - 0.5 * positive_variance) * dt
            + root_variance * sqrt_dt * stock_normal
        )
        variance = torch.clamp(initial_variance + factors.sum(dim=2), min=0.0)
        spots = torch.exp(log_spot)
        if statistic in {"spot_paths", "state_paths"}:
            values[:, :, step + 1] = spots
            if factor_paths is not None:
                factor_paths[:, :, step + 1, :] = factors
        elif statistic == "observation_spots":
            if (step + 1) % observation_stride == 0:
                values[:, :, (step + 1) // observation_stride - 1] = spots
        elif statistic in {"max_spot", "barrier_up"}:
            values = torch.maximum(values, spots)
        elif statistic == "barrier_down":
            values = torch.minimum(values, spots)
        elif statistic == "average_spot":
            values = values + spots
        elif statistic == "realized_volatility":
            values = values + (log_spot - previous_log_spot).square()

    terminal = torch.exp(log_spot)
    if statistic == "terminal_spot":
        values = terminal
    elif statistic == "average_spot":
        values = values / float(num_steps)
    elif statistic == "realized_volatility":
        values = torch.sqrt(52.0 / float(num_steps) * values)
    elif statistic in {"barrier_up", "barrier_down"}:
        values = torch.stack((terminal, values), dim=-1)
    if factor_paths is not None:
        return (values, factor_paths), torch.exp(-rate * maturity), spot0
    return values, torch.exp(-rate * maturity), spot0


def _statistic(name: PathStatistic):
    def run(*args, **kwargs):
        return simulate_statistic_batch(*args, **kwargs, statistic=name)
    return run


simulate_terminal_spot_batch = _statistic("terminal_spot")
simulate_max_spot_batch = _statistic("max_spot")
simulate_average_spot_batch = _statistic("average_spot")
simulate_realized_volatility_batch = _statistic("realized_volatility")
simulate_spot_paths_batch = _statistic("spot_paths")


def simulate_state_paths_batch(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    state, _, _ = simulate_statistic_batch(
        *args, **kwargs, statistic="state_paths"
    )
    return state


def simulate_observation_spots_batch(*args, **kwargs) -> torch.Tensor:
    values, _, _ = simulate_statistic_batch(
        *args, **kwargs, statistic="observation_spots"
    )
    return values


def simulate_barrier_state_batch(
    rows, model_by_id, product_by_id, num_paths, num_steps, device, dtype, up
) -> tuple[torch.Tensor, torch.Tensor]:
    state, _, _ = simulate_statistic_batch(
        rows, model_by_id, product_by_id, num_paths, num_steps, device, dtype,
        statistic="barrier_up" if up else "barrier_down",
    )
    return state[:, :, 0], state[:, :, 1]
