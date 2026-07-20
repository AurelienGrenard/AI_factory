"""Generate caplet contract terms."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.core_parameter_databases import write_product_database

DATABASE_ID = "caplets_01"
ROW_COUNT = 1_000
SEED = 741_500_101


def product_parameters() -> list[dict[str, float]]:
    fixing_draws = philox_uniforms(SEED, ROW_COUNT, stream=0)
    strike_draws = philox_uniforms(SEED, ROW_COUNT, stream=1)
    return [
        {
            "fixing_time": 0.25 * (
                1 + min(int(float(fixing_draws[index]) * 40), 39)
            ),
            "accrual_period": 0.5,
            "strike": 0.08 * float(strike_draws[index]),
            "notional": 1.0,
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    print(write_product_database(
        database_id=DATABASE_ID,
        product_family="caplets",
        title="Caplet production database",
        payoff={
            "expression": "N delta max(L(T;T,T+delta) - K, 0) paid at T+delta",
            "bond_option_equivalent": "N max(1 - (1 + delta K) P(T,T+delta), 0) at T",
        },
        parameters=product_parameters(),
        construction={
            "method": "random sample",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "rule": "Quarterly fixing dates, semiannual accrual, and uniform strike.",
            "bounds": {
                "fixing_time": [0.25, 10.0],
                "accrual_period": 0.5,
                "strike": [0.0, 0.08],
                "notional": 1.0,
            },
        },
        parameter_docs={
            "fixing_time": "Forward-rate fixing date T in years.",
            "accrual_period": "Year fraction delta from fixing to payment.",
            "strike": "Caplet strike K.",
            "notional": "Contract notional N.",
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
