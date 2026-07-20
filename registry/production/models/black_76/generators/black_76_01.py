"""Generate shifted Black-76 production parameters."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns

DATABASE_ID = "black_76_01"
ROW_COUNT = 1_000
SEED = 744_100_001
BOUNDS = {"volatility": (0.05, 0.80), "displacement": (0.03, 0.06)}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(
        seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS
    )
    return [
        {name: float(columns[name][index]) for name in BOUNDS}
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    print(write_model_database(
        database_id=DATABASE_ID,
        model_family="black_76",
        parameters=model_parameters(),
        title="Shifted Black-76 production model parameter database",
        construction={
            "method": "uniform random sample",
            "rule": "Black volatility and positive displacement are sampled independently.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "bounds": {key: list(value) for key, value in BOUNDS.items()},
        },
        parameter_docs={
            "volatility": "Annualized Black volatility sigma.",
            "displacement": "Positive shift s applied to the forward and strike.",
        },
        dynamics={
            "measure": "Product forward measure",
            "representation": "Shifted lognormal forward model",
            "equations": {
                "shifted_forward": "X_t = F_t + displacement",
                "dynamics": "dX_t / X_t = volatility dW_t",
            },
            "initial_conditions": {
                "forward": "F_0 implied by the aligned Nelson-Siegel curve"
            },
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
