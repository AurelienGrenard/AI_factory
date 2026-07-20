"""Batched CIR QE Bermudan swaption Monte Carlo."""

from __future__ import annotations

from time import perf_counter

import torch

from ai_factory.pytorch.common.bermudan_lsm import present_value_cashflows
from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import cir_bond_coefficients
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_CPU_BATCH_ROWS = 16
DEFAULT_CUDA_BATCH_ROWS = 64
MAX_EXERCISES = 8
MAX_PAYMENTS = 20
QE_PSI_CUTOFF = 1.5
BASIS_RATE_SCALE = 0.04


def _column(values, target):
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _price_batch_impl(rows, models, products, *, num_paths, target, target_dt, batch_rows):
    started = perf_counter()
    outputs = []
    payment_axis = torch.arange(MAX_PAYMENTS, device=target)[None, :]
    exercise_axis = torch.arange(MAX_EXERCISES, device=target)[None, :]
    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        ms = [models[row["model_id"]] for row in batch]
        ps = [products[row["product_id"]] for row in batch]
        size = len(batch)
        kappa = _column([m["kappa"] for m in ms], target)
        theta = _column([m["theta"] for m in ms], target)
        sigma = _column([m["volatility"] for m in ms], target)
        rate = _column([m["initial_rate"] for m in ms], target).expand(-1, num_paths).clone()
        first = _column([p["first_exercise"] for p in ps], target)
        period = _column([p["exercise_period"] for p in ps], target)
        accrual = _column([p["accrual_period"] for p in ps], target)
        strike = _column([p["fixed_rate"] for p in ps], target)
        notional = _column([p["notional"] for p in ps], target)
        direction = _column([p["direction"] for p in ps], target)
        exercise_count = torch.tensor([p["exercise_count"] for p in ps], device=target)
        payment_count = torch.tensor([p["payment_count"] for p in ps], device=target)
        exercise_times = first + period * exercise_axis
        schedule: dict[int, list[tuple[int, int]]] = {}
        for row_index, product in enumerate(ps):
            for exercise in range(product["exercise_count"]):
                step = round(
                    (product["first_exercise"] + exercise * product["exercise_period"])
                    / target_dt
                )
                schedule.setdefault(step, []).append((row_index, exercise))
        max_step = max(schedule)
        dt = target_dt
        decay = torch.exp(-kappa * dt)
        one_minus_decay = 1.0 - decay
        sigma_squared = sigma.square()
        integral = torch.zeros_like(rate)
        states = torch.zeros((size, num_paths, MAX_EXERCISES), device=target, dtype=torch.float64)
        discounts = torch.ones_like(states)
        for step in range(1, max_step + 1):
            previous = rate
            mean = theta + (rate - theta) * decay
            variance = (
                rate * sigma_squared * decay * one_minus_decay / kappa
                + theta * sigma_squared * one_minus_decay.square() / (2.0 * kappa)
            )
            psi = variance / mean.square()
            normal = torch.randn_like(rate)
            uniform = torch.rand_like(rate)
            inverse_psi = 1.0 / torch.clamp(psi, min=1.0e-14)
            b_squared = 2.0 * inverse_psi - 1.0 + torch.sqrt(2.0 * inverse_psi) * torch.sqrt(
                torch.clamp(2.0 * inverse_psi - 1.0, min=0.0)
            )
            quadratic = mean / (1.0 + b_squared) * (
                torch.sqrt(torch.clamp(b_squared, min=0.0)) + normal
            ).square()
            probability = (psi - 1.0) / (psi + 1.0)
            exponential = torch.where(
                uniform <= probability,
                torch.zeros_like(rate),
                mean / (1.0 - probability)
                * torch.log((1.0 - probability) / (1.0 - uniform)),
            )
            rate = torch.where(psi <= QE_PSI_CUTOFF, quadratic, exponential)
            integral += 0.5 * (previous + rate) * dt
            for row_index, exercise in schedule.get(step, ()):
                states[row_index, :, exercise] = rate[row_index]
                discounts[row_index, :, exercise] = torch.exp(-integral[row_index])

        immediate = torch.zeros_like(states)
        basis_state = torch.zeros_like(states)
        horizons = accrual[:, :, None] * (
            payment_axis[:, None, :] + 1 - exercise_axis[:, :, None]
        )
        bond_a, bond_b = cir_bond_coefficients(
            kappa[:, :, None], theta[:, :, None], sigma[:, :, None],
            torch.clamp(horizons, min=accrual[:, :, None]),
        )
        for exercise in range(MAX_EXERCISES):
            bonds = bond_a[:, exercise, :, None] * torch.exp(
                -bond_b[:, exercise, :, None] * states[:, None, :, exercise]
            )
            active = ((payment_axis >= exercise) & (payment_axis < payment_count[:, None]))[:, :, None]
            annuity = (active * accrual[:, :, None] * bonds).sum(dim=1)
            end_index = (payment_count - 1)[:, None, None].expand(-1, 1, num_paths)
            end_bond = torch.gather(bonds, 1, end_index).squeeze(1)
            immediate[:, :, exercise] = notional * torch.clamp(
                direction * (1.0 - end_bond - strike * annuity), min=0.0
            )
            basis_state[:, :, exercise] = (1.0 - end_bond) / annuity / BASIS_RATE_SCALE
        immediate *= (exercise_axis < exercise_count[:, None])[:, None, :]
        outputs.extend(price_summary(present_value_cashflows(
            immediate, basis_state, discounts, exercise_count
        )))
    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}


def price_batch(rows, model_by_id, product_by_id, *, num_paths, device,
                target_dt=1.0 / 52.0, batch_rows=None, **_):
    target = resolve_device(device)
    if batch_rows is None:
        batch_rows = (
            DEFAULT_CUDA_BATCH_ROWS
            if target.type == "cuda"
            else DEFAULT_CPU_BATCH_ROWS
        )
    if target.type == "cuda" and rows:
        warmup(target, shape=(min(batch_rows, len(rows)), num_paths), dtype=torch.float64)
        _price_batch_impl(
            rows[:batch_rows], model_by_id, product_by_id,
            num_paths=num_paths, target=target, target_dt=target_dt,
            batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows, model_by_id, product_by_id, num_paths=num_paths, target=target,
        target_dt=target_dt, batch_rows=batch_rows,
    )
