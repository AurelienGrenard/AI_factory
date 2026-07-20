"""Generate the lookback_fixed_calls_01 production product database."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root_from(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {path}")

import numpy as np

PROJECT_ROOT = _project_root_from(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.recipe_helpers import write_product_recipe_database

DATABASE_ID = "lookback_fixed_calls_01"
ROW_COUNT = 1_000
SEED = 730_000_001
MIN_MATURITY = 1.0 / 12.0
MAX_MATURITY = 3.0
LOG_MONEYNESS_SLOPE = 0.2


def product_parameters() -> list[dict[str, float]]:
    maturity_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=0)
    strike_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=1)

    maturities = MIN_MATURITY + (MAX_MATURITY - MIN_MATURITY) * maturity_uniforms
    log_strike_lower = -LOG_MONEYNESS_SLOPE * maturities
    log_strike_upper = LOG_MONEYNESS_SLOPE * maturities
    log_strikes = log_strike_lower + (
        log_strike_upper - log_strike_lower
    ) * strike_uniforms
    strikes = np.exp(log_strikes)

    return [
        {
            "strike": round(float(strikes[index]), 12),
            "maturity": round(float(maturities[index]), 12),
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    write_product_recipe_database(
        database_id=DATABASE_ID,
        product_family="lookback_fixed_calls",
        title="Fixed-strike lookback call production product parameter database",
        payoff_key="lookback_fixed_call",
        expression="max(max_t S_t - K, 0)",
        scaling_rule="V(s, K) = s * V(1, K / s)",
        parameters=product_parameters(),
        construction={
            "method": "random sample",
            "rule": "T is uniform on [1/12, 3]. Conditional on T, log(K) is uniform on [-aT, aT].",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "row_count": ROW_COUNT,
            "bounds": {
                "maturity": ["1/12", 3.0],
                "log_strike_given_maturity": "uniform on [-aT, aT]",
                "a": LOG_MONEYNESS_SLOPE,
            },
        },
        parameter_docs={
            "strike": "Strike in normalized spot units.",
            "maturity": "Maturity in years.",
        },
        registry_tier="production",
    )


if __name__ == "__main__":
    main()
