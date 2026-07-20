"""Metadata and time-grid conventions shared by result recipes."""

from __future__ import annotations

from typing import Any

DEFAULT_TARGET_DT = "1/52"
DEFAULT_RELATIVE_BUMP = 5.0e-4


def time_grid_documentation(target_dt: str | float | int) -> dict[str, str]:
    return {
        "rule": "nearest integer step count to target dt",
        "target_dt": str(target_dt),
        "step_count": "round(maturity / target_dt)",
        "effective_dt": "maturity / step_count",
    }


def price_only_outputs_documentation() -> dict[str, dict[str, str]]:
    return {
        "price": {"estimator": "Monte Carlo discounted payoff mean"},
        "standard_error": {
            "estimator": "Monte Carlo standard error of discounted payoff"
        },
    }


def delta_crn_outputs_documentation() -> dict[str, dict[str, str]]:
    return {
        **price_only_outputs_documentation(),
        "delta": {
            "estimator": "Central finite difference with common random numbers"
        },
        "delta_standard_error": {
            "estimator": "Monte Carlo standard error of pathwise finite-difference delta"
        },
    }


def delta_method_documentation(relative_bump: float) -> dict[str, Any]:
    return {
        "method": "central finite difference with common random numbers",
        "relative_bump": relative_bump,
        "spot_bumps": "S0 * (1 - relative_bump), S0, S0 * (1 + relative_bump)",
        "random_numbers": "identical Brownian drivers for base, down, and up paths",
    }
