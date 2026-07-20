"""Generate the heston_02 production model database.

This database mirrors heston_01 but uses strictly positive risk-free rates for
early-exercise products.
"""

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

from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns, uniform_between

DATABASE_ID = "heston_02"
ROW_COUNT = 1_000
SEED = 710_000_101

BOUNDS = {
    "kappa": (0.5, 4.0),
    "theta": (0.01, 0.15),
    "rho": (-1.0, -0.3),
    "initial_variance": (0.01, 0.12),
    "risk_free_rate": (0.01, 0.05),
}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS)
    kappa = columns["kappa"]
    theta = columns["theta"]
    gamma_lower = np.maximum(np.sqrt(kappa * theta / 5.0), 0.1)
    gamma_upper = np.minimum(np.sqrt(12.0 * kappa * theta), 0.8)
    gamma = uniform_between(
        seed=SEED,
        lower=gamma_lower,
        upper=gamma_upper,
        stream=len(BOUNDS),
    )
    return [
        {
            "spot": 1.0,
            "risk_free_rate": float(columns["risk_free_rate"][index]),
            "dividend_yield": 0.0,
            "initial_variance": float(columns["initial_variance"][index]),
            "theta": float(theta[index]),
            "kappa": float(kappa[index]),
            "volatility_of_variance": float(gamma[index]),
            "rho": float(columns["rho"][index]),
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    gamma_rule = (
        "gamma is sampled uniformly from "
        "[max(sqrt(kappa * theta / 5), 0.1), min(sqrt(12 * kappa * theta), 0.8)]."
    )
    path = write_model_database(
        database_id=DATABASE_ID,
        model_family="heston",
        parameters=model_parameters(),
        title=f"Heston production model parameter database {DATABASE_ID}",
        construction={
            "method": "random sample",
            "rule": (
                "kappa, theta, rho, initial_variance, and risk_free_rate are "
                "independent uniforms. "
                + gamma_rule
            ),
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "spot": 1.0,
            "dividend_yield": 0.0,
            "row_count": ROW_COUNT,
            "bounds": {
                "kappa": list(BOUNDS["kappa"]),
                "theta": list(BOUNDS["theta"]),
                "rho": list(BOUNDS["rho"]),
                "initial_variance": list(BOUNDS["initial_variance"]),
                "risk_free_rate": list(BOUNDS["risk_free_rate"]),
                "volatility_of_variance": gamma_rule,
            },
            "feller_ratio": {
                "expression": "2 * kappa * theta / volatility_of_variance^2",
                "guaranteed_range": [1.0 / 6.0, 10.0],
            },
        },
        parameter_docs={
            "spot": "Initial spot.",
            "risk_free_rate": "Continuously compounded risk-free rate.",
            "dividend_yield": "Continuously compounded dividend yield.",
            "initial_variance": "Initial variance v0.",
            "theta": "Long-run variance theta.",
            "kappa": "Variance mean-reversion speed kappa.",
            "volatility_of_variance": "Volatility of variance gamma.",
            "rho": "Spot/variance Brownian correlation rho.",
        },
        dynamics={
            "measure": "Risk-neutral",
            "equations": {
                "spot": "dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S",
                "variance": (
                    "dV_t = kappa (theta - V_t) dt "
                    "+ gamma sqrt(V_t) dW_t^V"
                ),
                "correlation": "d<W^S, W^V>_t = rho dt",
            },
            "initial_conditions": {
                "spot": "S_0 = spot",
                "variance": "V_0 = initial_variance",
            },
        },
        registry_tier="production",
    )
    print(path)


if __name__ == "__main__":
    main()
