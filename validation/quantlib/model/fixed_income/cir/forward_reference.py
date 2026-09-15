"""Independent CPU CIR checks: forward-law moments, PDE and held-out LSM.

The PDE works under the original risk-neutral measure, with the killing term
-r*V. It does not use the proposed forward transition or a regression.
No validation cache or production dataset is read or written by these functions.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import ncx2


def bond_coefficients(model: dict, maturity):
    k, theta, sigma = (model[key] for key in ('mean_reversion', 'long_term_mean', 'volatility'))
    gamma = math.hypot(k, math.sqrt(2) * sigma)
    difference = 2 * sigma * sigma / (gamma + k)
    one_minus = -np.expm1(-gamma * np.asarray(maturity))
    denominator = 2 * gamma - difference * one_minus
    log_a = 2 * k * theta / (sigma * sigma) * (
        -np.log1p(-difference * one_minus / (2 * gamma)) - .5 * difference * maturity)
    return log_a, 2 * one_minus / denominator


def bond(model: dict, rate, maturity):
    log_a, loading = bond_coefficients(model, maturity)
    return np.exp(log_a - loading * rate)


def forward_transition(model: dict, interval: float, remaining: float):
    """Return scale, degrees of freedom, and noncentrality per unit start rate."""
    k, theta, sigma = (model[key] for key in ('mean_reversion', 'long_term_mean', 'volatility'))
    gamma = math.hypot(k, math.sqrt(2) * sigma)
    # Direct published q/rho representation, distinct from the CUDA stable form.
    rho = 2 * gamma / (sigma * sigma * math.expm1(gamma * interval))
    q = 2 * (rho + (k + gamma) / (sigma * sigma) + bond_coefficients(model, remaining)[1])
    loading = 4 * rho * rho * math.exp(gamma * interval) / q
    return 1 / q, 4 * k * theta / (sigma * sigma), loading


def exercise_times(product: dict):
    return (product['first_exercise_time']
            + np.arange(product['exercise_count']) * product['payment_interval']) / 252


def exercise_payoff(model: dict, product: dict, rate, exercise: int, side='payer'):
    rates = np.asarray(rate)
    terms = np.arange(1, product['payment_count'] - exercise + 1) * product['payment_interval'] / 252
    log_a, loading = bond_coefficients(model, terms)
    annuity = np.zeros_like(rates, dtype=float)
    final_bond = np.ones_like(rates, dtype=float)
    for intercept, slope in zip(log_a, loading):
        final_bond = np.exp(intercept - slope * rates)
        annuity += final_bond
    swap = 1 - final_bond - product['strike'] * product['accrual_fraction'] * annuity
    return (product['notional'] * np.maximum(swap if side == 'payer' else -swap, 0)).reshape(rates.shape)


def pde_price(model: dict, product: dict, side='payer', nodes=768, steps_per_year=128,
              boundary_multiplier=1.0, european_only=False, payoff=None):
    """Monotone nonuniform spatial generator, CN/Rannacher between exercises.

    Grid and time convergence MUST be checked before treating a result as a
    reference. The upper boundary is reflecting and placed far in the tail.
    At zero the CIR drift gives the one-sided attainable-boundary generator.
    An optional transformed obstacle supports deterministic-shift extensions
    without changing this independent risk-neutral PDE operator.
    """
    k, theta, sigma, initial = (model[key] for key in
        ('mean_reversion', 'long_term_mean', 'volatility', 'initial_state'))
    times = exercise_times(product)
    variance = theta * sigma * sigma / (2 * k)
    degree = 4 * k * theta / (sigma * sigma)
    extrema = [theta + 12 * math.sqrt(variance), 3 * initial, .05]
    for time in times:
        decay = math.exp(-k * time)
        scale = sigma * sigma * -math.expm1(-k * time) / (4 * k)
        extrema.append(scale * ncx2.ppf(1 - 1e-11, degree, initial * decay / scale))
    maximum = boundary_multiplier * max(extrema)
    grid = maximum * np.linspace(0, 1, nodes + 1)**2
    left, right = np.zeros(nodes + 1), np.zeros(nodes + 1)
    hm, hp = np.diff(grid)[:-1], np.diff(grid)[1:]
    r = grid[1:-1]
    mu, diffusion = k * (theta - r), sigma * sigma * r
    left[1:-1] = (diffusion - mu * hp) / (hm * (hm + hp))
    right[1:-1] = (diffusion + mu * hm) / (hp * (hm + hp))
    negative = (left[1:-1] < 0) | (right[1:-1] < 0)
    left[1:-1][negative] = (diffusion / (hm * (hm + hp)) + np.maximum(-mu, 0) / hm)[negative]
    right[1:-1][negative] = (diffusion / (hp * (hm + hp)) + np.maximum(mu, 0) / hp)[negative]
    right[0] = k * theta / (grid[1] - grid[0])
    left[-1] = max(k * (grid[-1] - theta), 0) / (grid[-1] - grid[-2])
    diagonal = -left - right - grid

    def advance(values, interval):
        count = max(2, math.ceil(interval * steps_per_year))
        dt = interval / count
        # Two half implicit steps damp the kink introduced by exercise.
        for step in range(count + 1):
            weight, h = (1.0, dt / 2) if step < 2 else (.5, dt)
            rhs = values.copy()
            if weight < 1:
                rhs += (1 - weight) * h * diagonal * values
                rhs[1:] += (1 - weight) * h * left[1:] * values[:-1]
                rhs[:-1] += (1 - weight) * h * right[:-1] * values[1:]
            band = np.zeros((3, nodes + 1))
            band[0, 1:] = -weight * h * right[:-1]
            band[1] = 1 - weight * h * diagonal
            band[2, :-1] = -weight * h * left[1:]
            values = solve_banded((1, 1), band, rhs, check_finite=False)
        return values

    obstacle = payoff if payoff is not None else (
        lambda rate, exercise: exercise_payoff(model, product, rate, exercise, side)
    )
    values = obstacle(grid, len(times) - 1)
    for exercise in range(len(times) - 2, -1, -1):
        values = advance(values, times[exercise + 1] - times[exercise])
        if not european_only:
            values = np.maximum(values, obstacle(grid, exercise))
    values = advance(values, times[0])
    return float(np.interp(initial, grid, values))


def held_out_lsm(model: dict, product: dict, side='payer', paths=262144, seed=1):
    """Independent SciPy sampler/QR least squares; train and test paths are disjoint."""
    rng = np.random.default_rng(seed)
    times = exercise_times(product)
    terminal = times[-1]
    rates = np.full(2 * paths, model['initial_state'])
    observations = []
    previous = 0.0
    for time in times:
        scale, degree, loading = forward_transition(model, time - previous, terminal - time)
        rates = scale * rng.noncentral_chisquare(degree, loading * rates)
        observations.append(rates)
        previous = time
    initial_bond = float(bond(model, model['initial_state'], terminal))
    mean = model['long_term_mean']
    sd = math.sqrt(max(mean * model['volatility']**2 / (2 * model['mean_reversion']), 1e-12))

    def features(rate):
        x = (rate - mean) / sd
        return np.column_stack((np.ones_like(x), x, x*x-1, x*x*x-3*x))

    values = exercise_payoff(model, product, observations[-1], len(times)-1, side) * initial_bond
    coefficients = []
    for i in range(len(times)-2, -1, -1):
        immediate = exercise_payoff(model, product, observations[i], i, side)
        immediate *= initial_bond / bond(model, observations[i], terminal-times[i])
        candidate = immediate[:paths] > 0
        if candidate.sum() <= 4:
            coefficients.append(None)
            continue
        beta = np.linalg.lstsq(features(observations[i][:paths][candidate]), values[:paths][candidate], rcond=None)[0]
        continuation = features(observations[i]) @ beta
        values = np.where((immediate > 0) & (immediate > continuation), immediate, values)
        coefficients.append(beta.tolist())
    return {'training_price': float(values[:paths].mean()),
            'test_price': float(values[paths:].mean()),
            'test_standard_error': float(values[paths:].std(ddof=1) / math.sqrt(paths)),
            'training_standard_error': float(values[:paths].std(ddof=1) / math.sqrt(paths)),
            'paths_per_partition': paths, 'seed': seed, 'coefficients': coefficients}
