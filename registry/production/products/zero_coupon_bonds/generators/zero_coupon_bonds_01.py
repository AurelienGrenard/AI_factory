"""Generate zero-coupon bond terms over short and long maturities."""

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

DATABASE_ID = "zero_coupon_bonds_01"
ROW_COUNT = 1_000
SEED = 741_500_001


def product_parameters() -> list[dict[str, float]]:
    uniforms = philox_uniforms(SEED, ROW_COUNT)
    minimum = 1.0 / 12.0
    maximum = 30.0
    return [
        {"maturity": minimum + (maximum - minimum) * float(value), "notional": 1.0}
        for value in uniforms
    ]


def main() -> None:
    print(write_product_database(
        database_id=DATABASE_ID,
        product_family="zero_coupon_bonds",
        title="Zero-coupon bond production database",
        payoff={"expression": "N paid at maturity T", "price": "N P(0,T)"},
        parameters=product_parameters(),
        construction={
            "method": "random sample",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "rule": "Continuous uniform maturity from one month to thirty years.",
            "bounds": {"maturity": ["1/12", 30.0], "notional": 1.0},
        },
        parameter_docs={
            "maturity": "Payment date T in years.",
            "notional": "Amount N paid at maturity.",
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
