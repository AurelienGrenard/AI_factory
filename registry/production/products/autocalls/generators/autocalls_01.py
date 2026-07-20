"""Generate normalized quarterly memory autocall contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


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

DATABASE_ID = "autocalls_01"
ROW_COUNT = 1_000
SEED = 734_100_001


def product_parameters() -> list[dict[str, float | int]]:
    maturity_u = philox_uniforms(SEED, ROW_COUNT, stream=0)
    autocall_u = philox_uniforms(SEED, ROW_COUNT, stream=1)
    coupon_barrier_u = philox_uniforms(SEED, ROW_COUNT, stream=2)
    protection_u = philox_uniforms(SEED, ROW_COUNT, stream=3)
    coupon_u = philox_uniforms(SEED, ROW_COUNT, stream=4)
    maturity_quarters = 8 + np.floor(maturity_u * 13).astype(int)
    maturity_quarters = np.minimum(maturity_quarters, 20)
    autocall_barrier = 0.90 + 0.15 * autocall_u
    coupon_barrier = 0.65 + (np.minimum(0.90, autocall_barrier) - 0.65) * coupon_barrier_u
    protection_upper = np.minimum(0.75, coupon_barrier - 0.05)
    protection_barrier = 0.50 + (protection_upper - 0.50) * protection_u
    coupon_rate = 0.005 + 0.025 * coupon_u
    return [{
        "maturity": round(float(maturity_quarters[i]) / 4.0, 12),
        "autocall_barrier": round(float(autocall_barrier[i]), 12),
        "coupon_barrier": round(float(coupon_barrier[i]), 12),
        "protection_barrier": round(float(protection_barrier[i]), 12),
        "coupon_rate_per_observation": round(float(coupon_rate[i]), 12),
        "observation_count": int(maturity_quarters[i]),
        "first_autocall_observation": 4,
    } for i in range(ROW_COUNT)]


def main() -> None:
    write_product_recipe_database(
        database_id=DATABASE_ID,
        product_family="autocalls",
        title="Quarterly memory autocall production product parameter database",
        payoff_key="memory_autocall",
        expression=(
            "Quarterly memory coupons; redemption at the first eligible autocall date, "
            "otherwise max-protected nominal or S_T/S_0 at maturity."
        ),
        scaling_rule="V(N) = N * V(1)",
        parameters=product_parameters(),
        construction={
            "method": "random sample",
            "rule": "T is sampled on the quarterly grid [2, 5], with Bd < Bc <= Ba.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "row_count": ROW_COUNT,
            "bounds": {
                "maturity": [2.0, 5.0],
                "autocall_barrier": [0.90, 1.05],
                "coupon_barrier": "uniform on [0.65, min(0.90, Ba)]",
                "protection_barrier": "uniform on [0.50, min(0.75, Bc - 0.05)]",
                "coupon_rate_per_observation": [0.005, 0.03],
            },
            "observation_frequency": "quarterly",
            "first_autocall_observation": 4,
            "coupon_memory": True,
            "nominal": 1.0,
        },
        parameter_docs={
            "maturity": "Maturity in years.",
            "autocall_barrier": "Autocall barrier as a fraction of initial spot.",
            "coupon_barrier": "Coupon barrier as a fraction of initial spot.",
            "protection_barrier": "European downside protection barrier at maturity.",
            "coupon_rate_per_observation": "Coupon rate per quarterly observation.",
            "observation_count": "Number of quarterly observation dates.",
            "first_autocall_observation": "First observation eligible for autocall.",
        },
        registry_tier="production",
    )


if __name__ == "__main__":
    main()
