"""Generate the Hull-White 1-factor production model database."""

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

from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns

DATABASE_ID = "hull_white_01"
ROW_COUNT = 1_000
SEED = 740_200_001
BOUNDS = {
    "mean_reversion": (0.02, 0.50),
    "volatility": (0.002, 0.030),
}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS)
    return [
        {
            "mean_reversion": float(columns["mean_reversion"][index]),
            "volatility": float(columns["volatility"][index]),
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    print(
        write_model_database(
            database_id=DATABASE_ID,
            model_family="hull_white",
            parameters=model_parameters(),
            title="Hull White 1-factor production model parameter database",
            construction={
                "method": "random sample",
                "rule": "Mean reversion and short-rate volatility are independent uniforms.",
                "rng": "Project Philox-4x32-10 generator",
                "seed": SEED,
                "bounds": {key: list(value) for key, value in BOUNDS.items()},
            },
            parameter_docs={
                "mean_reversion": "Positive Ornstein-Uhlenbeck mean-reversion speed a.",
                "volatility": "Constant short-rate volatility sigma.",
            },
            dynamics={
                "measure": "Risk-neutral",
                "representation": "Shifted Ornstein-Uhlenbeck",
                "equations": {
                    "state": "dx_t = -a x_t dt + sigma dW_t",
                    "short_rate": "r_t = x_t + phi(t)",
                    "curve_shift": "phi(t) = f(0,t) + sigma^2 / (2 a^2) (1 - exp(-a t))^2",
                },
                "initial_conditions": {"state": "x_0 = 0"},
            },
            registry_tier="production",
        )
    )


if __name__ == "__main__":
    main()
