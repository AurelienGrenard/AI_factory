"""Generate the black_scholes_01 production model database."""

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

DATABASE_ID = "black_scholes_01"
ROW_COUNT = 1_000
SEED = 730_100_001

BOUNDS = {
    "volatility": (0.05, 0.50),
    "risk_free_rate": (0.00, 0.08),
    "dividend_yield": (0.00, 0.05),
}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS)
    return [
        {
            "spot": 1.0,
            "risk_free_rate": float(columns["risk_free_rate"][index]),
            "dividend_yield": float(columns["dividend_yield"][index]),
            "volatility": float(columns["volatility"][index]),
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    path = write_model_database(
        database_id=DATABASE_ID,
        model_family="black_scholes",
        parameters=model_parameters(),
        title=f"Black Scholes production model parameter database {DATABASE_ID}",
        construction={
            "method": "random sample",
            "rule": "volatility, risk_free_rate, and dividend_yield are independent uniforms.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "spot": 1.0,
            "row_count": ROW_COUNT,
            "bounds": {
                "volatility": list(BOUNDS["volatility"]),
                "risk_free_rate": list(BOUNDS["risk_free_rate"]),
                "dividend_yield": list(BOUNDS["dividend_yield"]),
            },
        },
        parameter_docs={
            "spot": "Initial spot.",
            "risk_free_rate": "Continuously compounded risk-free rate.",
            "dividend_yield": "Continuously compounded dividend yield.",
            "volatility": "Constant Black-Scholes volatility sigma.",
        },
        dynamics={
            "measure": "Risk-neutral",
            "equations": {
                "spot": "dS_t / S_t = (r - q) dt + sigma dW_t",
            },
            "initial_conditions": {
                "spot": "S_0 = spot",
            },
        },
        registry_tier="production",
    )
    print(path)


if __name__ == "__main__":
    main()
