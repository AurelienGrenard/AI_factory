"""Shared production recipes for discretely monitored barrier calls."""

from __future__ import annotations

from typing import Literal

import numpy as np

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.recipe_helpers import write_product_recipe_database

ROW_COUNT = 1_000
SEED = 735_000_001
MIN_MATURITY = 1.0 / 12.0
MAX_MATURITY = 3.0
LOG_MONEYNESS_SCALE = 0.30


def product_parameters(direction: Literal["down", "up"]) -> list[dict[str, float | int]]:
    maturity_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=0)
    strike_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=1)
    barrier_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=2 if direction == "down" else 3)
    maturities = MIN_MATURITY + (MAX_MATURITY - MIN_MATURITY) * maturity_uniforms
    log_width = LOG_MONEYNESS_SCALE * np.sqrt(maturities)
    strikes = np.exp(-log_width + 2.0 * log_width * strike_uniforms)
    if direction == "down":
        barriers = 0.55 + 0.40 * barrier_uniforms
    else:
        lower = np.maximum(1.05, strikes + 0.05)
        upper = np.maximum(1.50, strikes + 0.10)
        barriers = lower + (upper - lower) * barrier_uniforms
    return [
        {
            "strike": round(float(strikes[index]), 12),
            "barrier": round(float(barriers[index]), 12),
            "maturity": round(float(maturities[index]), 12),
            "observation_count": max(1, round(52.0 * float(maturities[index]))),
            "rebate": 0.0,
        }
        for index in range(ROW_COUNT)
    ]


def write_barrier_call_database(direction: Literal["down", "up"], knock: Literal["in", "out"]) -> None:
    family = f"{direction}_and_{knock}_calls"
    hit = "min_i S(t_i) <= B" if direction == "down" else "max_i S(t_i) >= B"
    active = hit if knock == "in" else f"not ({hit})"
    write_product_recipe_database(
        database_id=f"{family}_01",
        product_family=family,
        title=f"{direction.title()}-and-{knock} call production product parameter database",
        payoff_key=f"{direction}_and_{knock}_call",
        expression=f"1_{{{active}}} * max(S_T - K, 0)",
        scaling_rule="V(s, K, B) = s * V(1, K / s, B / s)",
        parameters=product_parameters(direction),
        construction={
            "method": "random sample",
            "rule": "T is uniform on [1/12, 3], log(K) is uniform on [-a sqrt(T), a sqrt(T)], and the barrier is sampled conditionally on its direction.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "row_count": ROW_COUNT,
            "bounds": {
                "maturity": ["1/12", 3.0],
                "log_strike_given_maturity": "uniform on [-a sqrt(T), a sqrt(T)]",
                "a": LOG_MONEYNESS_SCALE,
                "barrier": [0.55, 0.95] if direction == "down" else "uniform on [max(1.05, K + 0.05), max(1.50, K + 0.10)]",
            },
        },
        parameter_docs={
            "strike": "Strike in normalized spot units.",
            "barrier": "Discrete barrier in normalized spot units.",
            "maturity": "Maturity in years.",
            "observation_count": "Number of equally spaced monitoring dates, round(52T).",
            "rebate": "Cash rebate, fixed to zero for this database.",
        },
        registry_tier="production",
    )
