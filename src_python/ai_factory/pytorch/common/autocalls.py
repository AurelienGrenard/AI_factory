"""Shared vectorized autocall payoff evaluation."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Callable

import torch

from ai_factory.pytorch.common.device import synchronize
from ai_factory.pytorch.common.pathwise_products import parameter_tensor

ObservationSimulator = Callable[..., torch.Tensor]


def price_from_observations(
    observation_spots: torch.Tensor,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    include_diagnostics: bool = True,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Price memory autocalls and return per-row economic diagnostics."""

    device = observation_spots.device
    dtype = observation_spots.dtype
    row_count, num_paths, observation_count = observation_spots.shape
    spot0 = parameter_tensor(
        rows, model_by_id, product_by_id, "spot", source="model", device=device, dtype=dtype
    )
    rate = parameter_tensor(
        rows, model_by_id, product_by_id, "risk_free_rate", source="model", device=device, dtype=dtype
    )
    maturity = parameter_tensor(
        rows, model_by_id, product_by_id, "maturity", source="product", device=device, dtype=dtype
    )
    autocall_barrier = parameter_tensor(
        rows, model_by_id, product_by_id, "autocall_barrier", source="product", device=device, dtype=dtype
    )
    coupon_barrier = parameter_tensor(
        rows, model_by_id, product_by_id, "coupon_barrier", source="product", device=device, dtype=dtype
    )
    protection_barrier = parameter_tensor(
        rows, model_by_id, product_by_id, "protection_barrier", source="product", device=device, dtype=dtype
    )
    coupon_rate = parameter_tensor(
        rows, model_by_id, product_by_id, "coupon_rate_per_observation", source="product", device=device, dtype=dtype
    )
    first_call = torch.tensor(
        [int(product_by_id[row["product_id"]]["first_autocall_observation"]) for row in rows],
        device=device,
        dtype=torch.long,
    )

    performance = observation_spots / spot0.view(-1, 1, 1)
    coupon_eligible = performance >= coupon_barrier.view(-1, 1, 1)
    observation_indices = torch.arange(
        1, observation_count + 1, device=device, dtype=torch.long
    ).view(1, 1, -1)
    callable_dates = observation_indices >= first_call.view(-1, 1, 1)
    call_hits = callable_dates & (
        performance >= autocall_barrier.view(-1, 1, 1)
    )
    has_call = call_hits.any(dim=2)
    first_call_zero_based = call_hits.to(torch.int8).argmax(dim=2)
    call_observation = first_call_zero_based + 1
    active = (~has_call).unsqueeze(2) | (
        observation_indices <= call_observation.unsqueeze(2)
    )

    eligible_indices = torch.where(
        coupon_eligible,
        observation_indices.expand(row_count, num_paths, -1),
        torch.zeros((), device=device, dtype=torch.long),
    )
    previous_paid = torch.cummax(eligible_indices, dim=2).values
    previous_paid = torch.nn.functional.pad(previous_paid[:, :, :-1], (1, 0))
    coupons_due = (observation_indices - previous_paid).to(dtype)
    coupon_cash = (
        coupon_eligible.to(dtype)
        * active.to(dtype)
        * coupons_due
        * coupon_rate.view(-1, 1, 1)
    )
    times = maturity.view(-1, 1, 1) * observation_indices.to(dtype) / float(
        observation_count
    )
    discounted_coupons = coupon_cash * torch.exp(-rate.view(-1, 1, 1) * times)
    coupon_pv = discounted_coupons.sum(dim=2)

    call_time = maturity.view(-1, 1) * call_observation.to(dtype) / float(
        observation_count
    )
    call_redemption = torch.exp(-rate.view(-1, 1) * call_time)
    terminal_performance = performance[:, :, -1]
    loss = (~has_call) & (
        terminal_performance < protection_barrier.view(-1, 1)
    )
    maturity_redemption = torch.where(
        loss,
        terminal_performance,
        torch.ones((), device=device, dtype=dtype),
    ) * torch.exp(-rate.view(-1, 1) * maturity.view(-1, 1))
    discounted_payoff = coupon_pv + torch.where(
        has_call, call_redemption, maturity_redemption
    )

    prices = discounted_payoff.mean(dim=1)
    stderrs = discounted_payoff.std(dim=1, unbiased=True) / math.sqrt(num_paths)
    output_values = torch.stack((prices, stderrs), dim=1).detach().cpu().tolist()
    outputs = [
        {"price": values[0], "standard_error": values[1]}
        for values in output_values
    ]
    if not include_diagnostics:
        return outputs, []

    autocall_probability = has_call.to(dtype).mean(dim=1)
    call_time_sum = torch.where(has_call, call_time, torch.zeros_like(call_time)).sum(dim=1)
    call_count = has_call.sum(dim=1)
    mean_call_time = torch.where(
        call_count > 0,
        call_time_sum / torch.clamp(call_count.to(dtype), min=1.0),
        torch.zeros_like(call_time_sum),
    )
    coupon_frequency = (coupon_eligible & active).to(dtype).sum(dim=(1, 2)) / (
        float(num_paths) * float(observation_count)
    )
    total_coupon = coupon_cash.sum(dim=2).mean(dim=1)
    loss_probability = loss.to(dtype).mean(dim=1)
    loss_count = loss.sum(dim=1)
    loss_redemption = torch.where(
        loss, terminal_performance, torch.zeros_like(terminal_performance)
    ).sum(dim=1)
    mean_loss_redemption = torch.where(
        loss_count > 0,
        loss_redemption / torch.clamp(loss_count.to(dtype), min=1.0),
        torch.zeros_like(loss_redemption),
    )

    diagnostic_values = torch.stack(
        (
            autocall_probability,
            mean_call_time,
            1.0 - autocall_probability,
            coupon_frequency,
            total_coupon,
            loss_probability,
            mean_loss_redemption,
        ),
        dim=1,
    ).detach().cpu().tolist()
    diagnostics = [
        {
            "autocall_probability": values[0],
            "mean_autocall_time": values[1],
            "maturity_probability": values[2],
            "coupon_payment_frequency": values[3],
            "mean_total_coupon": values[4],
            "capital_loss_probability": values[5],
            "mean_redemption_given_loss": values[6],
        }
        for values in diagnostic_values
    ]
    return outputs, diagnostics


@torch.inference_mode()
def price_batch(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    num_steps: int,
    device: str,
    simulate_observations: ObservationSimulator,
    dtype: torch.dtype = torch.float64,
    batch_rows: int = 8,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    target = torch.device(device)
    synchronize(target)
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_rows):
        chunk = rows[start : start + batch_rows]
        observation_spots = simulate_observations(
            chunk,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=target,
            dtype=dtype,
        )
        chunk_outputs, _ = price_from_observations(
            observation_spots,
            chunk,
            model_by_id,
            product_by_id,
            include_diagnostics=False,
        )
        outputs.extend(chunk_outputs)
    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}
