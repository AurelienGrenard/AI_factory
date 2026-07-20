"""Generate economically meaningful Bermudan swaption terms."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.product.core_parameter_databases import write_product_database

DATABASE_ID = "bermudan_swaptions_01"
ROW_COUNT = 1_000
SEED = 741_400_101


def product_parameters() -> list[dict[str, float | int]]:
    uniforms = [philox_uniforms(SEED, ROW_COUNT, stream=i) for i in range(5)]
    parameters: list[dict[str, float | int]] = []
    for index in range(ROW_COUNT):
        payment_count = 4 + min(int(float(uniforms[1][index]) * 17), 16)
        max_exercises = min(8, payment_count - 2)
        exercise_count = 2 + min(
            int(float(uniforms[2][index]) * (max_exercises - 1)),
            max_exercises - 2,
        )
        parameters.append(
            {
                "first_exercise": 0.5
                * (1 + min(int(float(uniforms[0][index]) * 10), 9)),
                "exercise_period": 0.5,
                "exercise_count": exercise_count,
                "accrual_period": 0.5,
                "payment_count": payment_count,
                "fixed_rate": 0.08 * float(uniforms[3][index]),
                "direction": 1 if float(uniforms[4][index]) < 0.5 else -1,
                "notional": 1.0,
            }
        )
    return parameters


def main() -> None:
    print(
        write_product_database(
            database_id=DATABASE_ID,
            product_family="bermudan_swaptions",
            title="Bermudan swaption production database",
            payoff={
                "exercise_value": "N max(direction [1 - P(t,Tn) - K delta sum_i P(t,Ti)], 0)",
                "exercise_rule": "At exercise j, enter the remaining swap with payment_count - j fixed-leg payments.",
                "direction": "+1 payer swaption, -1 receiver swaption",
            },
            parameters=product_parameters(),
            construction={
                "method": "random sample",
                "rng": "Project Philox-4x32-10 generator",
                "seed": SEED,
                "rule": "Semiannual exercises and payments; at least two fixed coupons remain after the last exercise.",
                "bounds": {
                    "first_exercise": [0.5, 5.0],
                    "exercise_period": 0.5,
                    "exercise_count": [2, 8],
                    "accrual_period": 0.5,
                    "payment_count": [4, 20],
                    "fixed_rate": [0.0, 0.08],
                    "direction": [-1, 1],
                    "notional": 1.0,
                },
            },
            parameter_docs={
                "first_exercise": "First exercise date in years.",
                "exercise_period": "Year fraction between exercise opportunities.",
                "exercise_count": "Number of Bermudan exercise opportunities.",
                "accrual_period": "Underlying fixed-leg year fraction.",
                "payment_count": "Fixed-leg payments remaining at first exercise.",
                "fixed_rate": "Underlying fixed rate K.",
                "direction": "+1 payer or -1 receiver swaption.",
                "notional": "Contract notional N.",
            },
            registry_tier="production",
        )
    )


if __name__ == "__main__":
    main()
