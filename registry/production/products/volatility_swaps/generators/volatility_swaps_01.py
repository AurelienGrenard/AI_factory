"""Generate the volatility_swaps_01 production product database."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root_from(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {path}")

PROJECT_ROOT = _project_root_from(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.recipe_helpers import write_product_recipe_database

DATABASE_ID = "volatility_swaps_01"
ROW_COUNT = 1_000
SEED = 733_000_001
MIN_MATURITY = 1.0 / 12.0
MAX_MATURITY = 3.0
MIN_VOLATILITY_STRIKE = 0.05
MAX_VOLATILITY_STRIKE = 0.45
OBSERVATION_FREQUENCY = "weekly"
OBSERVATIONS_PER_YEAR = 52


def product_parameters() -> list[dict[str, float | str | int]]:
    maturity_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=0)
    strike_uniforms = philox_uniforms(SEED, ROW_COUNT, stream=1)

    maturities = MIN_MATURITY + (MAX_MATURITY - MIN_MATURITY) * maturity_uniforms
    strikes = MIN_VOLATILITY_STRIKE + (
        MAX_VOLATILITY_STRIKE - MIN_VOLATILITY_STRIKE
    ) * strike_uniforms

    return [
        {
            "volatility_strike": round(float(strikes[index]), 12),
            "maturity": round(float(maturities[index]), 12),
            "observation_frequency": OBSERVATION_FREQUENCY,
            "observations_per_year": OBSERVATIONS_PER_YEAR,
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    write_product_recipe_database(
        database_id=DATABASE_ID,
        product_family="Volatility Swap",
        title="Volatility swap production product parameter database",
        payoff_key="volatility_swap",
        expression="sqrt(52 / N * sum_i log(S_i / S_{i-1})^2) - K_vol",
        scaling_rule="V(s, K_vol) = V(1, K_vol)",
        parameters=product_parameters(),
        construction={
            "row_count": ROW_COUNT,
            "method": "random sample",
            "rule": "T is uniform on [1/12, 3]. K_vol is uniform on [0.05, 0.45]. Observation frequency is weekly.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "bounds": {
                "maturity": ["1/12", 3.0],
                "volatility_strike": [MIN_VOLATILITY_STRIKE, MAX_VOLATILITY_STRIKE],
                "observation_frequency": OBSERVATION_FREQUENCY,
                "observations_per_year": OBSERVATIONS_PER_YEAR,
            },
        },
        parameter_docs={
            "volatility_strike": "Annualized realized-volatility strike.",
            "maturity": "Maturity in years.",
            "observation_frequency": "Fixed weekly observation convention.",
            "observations_per_year": "Annualization factor for weekly log-return observations.",
        },
        registry_tier="production",
    )


if __name__ == "__main__":
    main()
