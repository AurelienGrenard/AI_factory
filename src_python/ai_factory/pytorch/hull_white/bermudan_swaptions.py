"""Batched exact-transition Hull-White Bermudan swaption Monte Carlo."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.bermudan_lsm import present_value_cashflows
from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import (
    hull_white_bond_coefficients,
    hull_white_integral_variance,
    hull_white_state_integral_covariance,
    hull_white_state_variance,
    nelson_siegel_discount,
)
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_CPU_BATCH_ROWS = 32
DEFAULT_CUDA_BATCH_ROWS = 128
MAX_EXERCISES = 8
MAX_PAYMENTS = 20
BASIS_RATE_SCALE = 0.04


def _column(values: list[float], target: torch.device) -> torch.Tensor:
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _price_batch_impl(rows, models, curves, products, *, num_paths, target, batch_rows):
    started = perf_counter()
    outputs: list[dict[str, float]] = []
    payment_axis = torch.arange(MAX_PAYMENTS, device=target)[None, :]
    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        ms = [models[row["model_id"]] for row in batch]
        cs = [curves[row["curve_id"]] for row in batch]
        ps = [products[row["product_id"]] for row in batch]
        batch_size = len(batch)
        a = _column([m["mean_reversion"] for m in ms], target)
        sigma = _column([m["volatility"] for m in ms], target)
        beta0 = _column([c["beta0"] for c in cs], target)
        beta1 = _column([c["beta1"] for c in cs], target)
        beta2 = _column([c["beta2"] for c in cs], target)
        tau = _column([c["tau"] for c in cs], target)
        first = _column([p["first_exercise"] for p in ps], target)
        period = _column([p["exercise_period"] for p in ps], target)
        accrual = _column([p["accrual_period"] for p in ps], target)
        strike = _column([p["fixed_rate"] for p in ps], target)
        notional = _column([p["notional"] for p in ps], target)
        direction = _column([p["direction"] for p in ps], target)
        exercise_count = torch.tensor(
            [p["exercise_count"] for p in ps], device=target, dtype=torch.long
        )
        payment_count = torch.tensor(
            [p["payment_count"] for p in ps], device=target, dtype=torch.long
        )
        exercise_axis = torch.arange(MAX_EXERCISES, device=target)[None, :]
        exercise_times = first + period * exercise_axis
        exercise_active = exercise_axis < exercise_count[:, None]

        bond_a = torch.zeros(
            (batch_size, MAX_EXERCISES, MAX_PAYMENTS),
            device=target,
            dtype=torch.float64,
        )
        bond_b = torch.zeros_like(bond_a)
        for exercise in range(MAX_EXERCISES):
            maturities = first + accrual * (payment_axis + 1)
            coefficients = hull_white_bond_coefficients(
                exercise_times[:, exercise : exercise + 1],
                maturities,
                a,
                sigma,
                beta0,
                beta1,
                beta2,
                tau,
            )
            bond_a[:, exercise, :] = coefficients[0]
            bond_b[:, exercise, :] = coefficients[1]

        state = torch.zeros((batch_size, num_paths), device=target, dtype=torch.float64)
        stochastic_integral = torch.zeros_like(state)
        states = torch.zeros(
            (batch_size, num_paths, MAX_EXERCISES), device=target, dtype=torch.float64
        )
        discounts = torch.ones_like(states)
        previous_time = torch.zeros_like(first)
        for exercise in range(MAX_EXERCISES):
            interval = exercise_times[:, exercise : exercise + 1] - previous_time
            normals = torch.randn(
                (batch_size, num_paths, 2), device=target, dtype=torch.float64
            )
            decay = torch.exp(-a * interval)
            state_variance = hull_white_state_variance(a, sigma, interval)
            integral_variance = hull_white_integral_variance(a, sigma, interval)
            covariance = hull_white_state_integral_covariance(a, sigma, interval)
            state_scale = torch.sqrt(state_variance)
            loading = covariance / state_scale
            previous_state = state
            state = decay * state + state_scale * normals[:, :, 0]
            stochastic_integral = (
                stochastic_integral
                + (-torch.expm1(-a * interval) / a) * previous_state
                + loading * normals[:, :, 0]
                + torch.sqrt(torch.clamp(integral_variance - loading.square(), min=0.0))
                * normals[:, :, 1]
            )
            deterministic_integral = (
                -torch.log(
                    nelson_siegel_discount(
                        exercise_times[:, exercise : exercise + 1],
                        beta0,
                        beta1,
                        beta2,
                        tau,
                    )
                )
                + 0.5
                * hull_white_integral_variance(
                    a, sigma, exercise_times[:, exercise : exercise + 1]
                )
            )
            states[:, :, exercise] = state
            discounts[:, :, exercise] = torch.exp(
                -deterministic_integral - stochastic_integral
            )
            previous_time = exercise_times[:, exercise : exercise + 1]

        immediate = torch.zeros_like(states)
        basis_state = torch.zeros_like(states)
        for exercise in range(MAX_EXERCISES):
            bonds = bond_a[:, exercise, :, None] * torch.exp(
                -bond_b[:, exercise, :, None] * states[:, None, :, exercise]
            )
            active_payment = (
                (payment_axis >= exercise) & (payment_axis < payment_count[:, None])
            )[:, :, None]
            annuity = (active_payment * accrual[:, :, None] * bonds).sum(dim=1)
            end_index = (payment_count - 1)[:, None, None].expand(-1, 1, num_paths)
            end_bond = torch.gather(bonds, 1, end_index).squeeze(1)
            signed_swap = direction * (1.0 - end_bond - strike * annuity)
            immediate[:, :, exercise] = notional * torch.clamp(signed_swap, min=0.0)
            basis_state[:, :, exercise] = (1.0 - end_bond) / annuity / BASIS_RATE_SCALE
        immediate *= exercise_active[:, None, :]
        cashflows = present_value_cashflows(
            immediate, basis_state, discounts, exercise_count
        )
        outputs.extend(price_summary(cashflows))
    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}


def price_batch(rows, model_by_id, curve_by_id, product_by_id, *, num_paths, device,
                batch_rows=None, **_):
    target = resolve_device(device)
    if batch_rows is None:
        batch_rows = (
            DEFAULT_CUDA_BATCH_ROWS
            if target.type == "cuda"
            else DEFAULT_CPU_BATCH_ROWS
        )
    if target.type == "cuda" and rows:
        warmup(target, shape=(min(batch_rows, len(rows)), num_paths, 2), dtype=torch.float64)
        _price_batch_impl(
            rows[:batch_rows], model_by_id, curve_by_id, product_by_id,
            num_paths=num_paths, target=target, batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows, model_by_id, curve_by_id, product_by_id,
        num_paths=num_paths, target=target, batch_rows=batch_rows,
    )
