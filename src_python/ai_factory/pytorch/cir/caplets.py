"""Batched CIR QE caplet Monte Carlo."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

import torch

from ai_factory.pytorch.common.device import resolve_device, synchronize, warmup
from ai_factory.pytorch.common.fixed_income import cir_bond_coefficients
from ai_factory.pytorch.common.monte_carlo import price_summary

DEFAULT_CPU_BATCH_ROWS = 32
DEFAULT_CUDA_BATCH_ROWS = 64
QE_PSI_CUTOFF = 1.5


def _column(values, target):
    return torch.tensor(values, device=target, dtype=torch.float64)[:, None]


def _step_count(product, target_dt):
    return max(1, round(float(product["fixing_time"]) / target_dt))


def _price_group(rows, models, products, *, num_paths, num_steps, target_dt,
                 target, batch_rows):
    outputs = []
    for offset in range(0, len(rows), batch_rows):
        batch = rows[offset : offset + batch_rows]
        ms = [models[row["model_id"]] for row in batch]
        ps = [products[row["product_id"]] for row in batch]
        kappa = _column([model["kappa"] for model in ms], target)
        theta = _column([model["theta"] for model in ms], target)
        volatility = _column([model["volatility"] for model in ms], target)
        initial_rate = _column([model["initial_rate"] for model in ms], target)
        fixing = _column([product["fixing_time"] for product in ps], target)
        accrual = _column([product["accrual_period"] for product in ps], target)
        strike = _column([product["strike"] for product in ps], target)
        notional = _column([product["notional"] for product in ps], target)
        bond_a, bond_b = cir_bond_coefficients(kappa, theta, volatility, accrual)

        rate = initial_rate.expand(-1, num_paths).clone()
        integral = torch.zeros_like(rate)
        if num_steps is None:
            host_counts = [_step_count(product, target_dt) for product in ps]
            max_steps = max(host_counts)
            step_counts = torch.tensor(host_counts, device=target)[:, None]
        else:
            max_steps = num_steps
            step_counts = torch.full((len(batch), 1), num_steps, device=target)
        dt = fixing / step_counts
        decay = torch.exp(-kappa * dt)
        one_minus_decay = 1.0 - decay
        volatility_squared = volatility.square()
        for step in range(max_steps):
            previous = rate
            mean = theta + (rate - theta) * decay
            variance = (
                rate * volatility_squared * decay * one_minus_decay / kappa
                + theta * volatility_squared * one_minus_decay.square() / (2.0 * kappa)
            )
            psi = variance / mean.square()
            normal = torch.randn_like(rate)
            uniform = torch.rand_like(rate)
            inverse_psi = 1.0 / torch.clamp(psi, min=1.0e-14)
            b_squared = 2.0 * inverse_psi - 1.0 + torch.sqrt(
                2.0 * inverse_psi * torch.clamp(2.0 * inverse_psi - 1.0, min=0.0)
            )
            quadratic = mean / (1.0 + b_squared) * (
                torch.sqrt(torch.clamp(b_squared, min=0.0)) + normal
            ).square()
            probability = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - probability) / mean
            exponential = torch.where(
                uniform <= probability,
                torch.zeros_like(rate),
                torch.log((1.0 - probability) / (1.0 - uniform)) / beta,
            )
            proposed = torch.where(psi <= QE_PSI_CUTOFF, quadratic, exponential)
            active = step < step_counts
            rate = torch.where(active, proposed, previous)
            integral += torch.where(active, 0.5 * (previous + proposed) * dt, 0.0)
        bond = bond_a * torch.exp(-bond_b * rate)
        payoff = torch.exp(-integral) * notional * torch.clamp(
            1.0 - (1.0 + accrual * strike) * bond, min=0.0
        )
        outputs.extend(price_summary(payoff))
    return outputs


def _price_batch_impl(rows, models, products, *, num_paths, target, target_dt,
                      batch_rows):
    started = perf_counter()
    if target.type == "cuda":
        outputs = _price_group(
            rows, models, products, num_paths=num_paths, num_steps=None,
            target_dt=target_dt, target=target, batch_rows=batch_rows,
        )
    else:
        grouped = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[_step_count(products[row["product_id"]], target_dt)].append((index, row))
        ordered = [None] * len(rows)
        for steps, indexed_rows in grouped.items():
            group_outputs = _price_group(
                [row for _, row in indexed_rows], models, products,
                num_paths=num_paths, num_steps=steps, target_dt=target_dt,
                target=target, batch_rows=batch_rows,
            )
            for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
                ordered[index] = output
        outputs = [output for output in ordered if output is not None]
    synchronize(target)
    return outputs, {"wall_seconds": perf_counter() - started}


def price_batch(rows, model_by_id, product_by_id, *, num_paths, device,
                target_dt=1.0 / 52.0, batch_rows=None, **_: Any):
    target = resolve_device(device)
    batch_rows = batch_rows or (
        DEFAULT_CUDA_BATCH_ROWS if target.type == "cuda" else DEFAULT_CPU_BATCH_ROWS
    )
    if target.type == "cuda" and rows:
        warmup(target, shape=(min(batch_rows, len(rows)), num_paths), dtype=torch.float64)
        _price_batch_impl(
            rows[:batch_rows], model_by_id, product_by_id, num_paths=num_paths,
            target=target, target_dt=target_dt, batch_rows=batch_rows,
        )
    return _price_batch_impl(
        rows, model_by_id, product_by_id, num_paths=num_paths, target=target,
        target_dt=target_dt, batch_rows=batch_rows,
    )
