"""Generate European swaption terms."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.core_parameter_databases import write_product_database

DATABASE_ID = "swaptions_01"
ROW_COUNT = 1_000
SEED = 741_300_101


def product_parameters() -> list[dict[str, float | int]]:
    u = [philox_uniforms(SEED, ROW_COUNT, stream=i) for i in range(4)]
    return [{
        "expiry": 0.25 * (1 + min(int(float(u[0][i]) * 20), 19)),
        "accrual_period": 0.5,
        "payment_count": 2 + min(int(float(u[1][i]) * 19), 18),
        "fixed_rate": 0.08 * float(u[2][i]),
        "direction": 1 if float(u[3][i]) < 0.5 else -1,
        "notional": 1.0,
    } for i in range(ROW_COUNT)]


def main() -> None:
    print(write_product_database(
        database_id=DATABASE_ID, product_family="swaptions",
        title="European swaption production database",
        payoff={
            "expression": "N max(direction [1 - P(T0,Tn) - K delta sum_i P(T0,Ti)], 0) at T0",
            "direction": "+1 payer swaption, -1 receiver swaption",
        },
        parameters=product_parameters(),
        construction={
            "method": "random sample", "rng": "Project Philox-4x32-10 generator", "seed": SEED,
            "rule": "Quarterly expiries, semiannual underlying swaps, 1Y-10Y tenors, uniform strike, balanced payer/receiver direction.",
            "bounds": {"expiry": [0.25, 5.0], "accrual_period": 0.5, "payment_count": [2, 20], "fixed_rate": [0.0, 0.08], "direction": [-1, 1], "notional": 1.0},
        },
        parameter_docs={
            "expiry": "Option expiry and underlying swap start T0.", "accrual_period": "Underlying fixed-leg year fraction delta.",
            "payment_count": "Number of underlying fixed-leg payments.", "fixed_rate": "Underlying fixed rate K.",
            "direction": "+1 payer or -1 receiver swaption.", "notional": "Contract notional N.",
        }, registry_tier="production",
    ))


if __name__ == "__main__":
    main()
