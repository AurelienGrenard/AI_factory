"""Generate Heston parameters with variable rates for autocall production."""

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

from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns, uniform_between

DATABASE_ID = "heston_03"
ROW_COUNT = 1_000
SEED = 710_000_201
BOUNDS = {
    "kappa": (0.5, 4.0),
    "theta": (0.01, 0.15),
    "rho": (-1.0, -0.3),
    "initial_variance": (0.01, 0.12),
    "risk_free_rate": (0.0, 0.08),
    "dividend_yield": (0.0, 0.05),
}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS)
    kappa = columns["kappa"]
    theta = columns["theta"]
    gamma_lower = np.maximum(np.sqrt(kappa * theta / 5.0), 0.1)
    gamma_upper = np.minimum(np.sqrt(12.0 * kappa * theta), 0.8)
    gamma = uniform_between(
        seed=SEED, lower=gamma_lower, upper=gamma_upper, stream=len(BOUNDS)
    )
    return [{
        "spot": 1.0,
        "risk_free_rate": float(columns["risk_free_rate"][i]),
        "dividend_yield": float(columns["dividend_yield"][i]),
        "initial_variance": float(columns["initial_variance"][i]),
        "theta": float(theta[i]),
        "kappa": float(kappa[i]),
        "volatility_of_variance": float(gamma[i]),
        "rho": float(columns["rho"][i]),
    } for i in range(ROW_COUNT)]


def main() -> None:
    gamma_rule = (
        "gamma is uniform on [max(sqrt(kappa * theta / 5), 0.1), "
        "min(sqrt(12 * kappa * theta), 0.8)]."
    )
    print(write_model_database(
        database_id=DATABASE_ID,
        model_family="heston",
        parameters=model_parameters(),
        title=f"Heston production model parameter database {DATABASE_ID}",
        construction={
            "method": "random sample",
            "rule": "kappa, theta, rho, v0, r, and q are independent uniforms; " + gamma_rule,
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "spot": 1.0,
            "row_count": ROW_COUNT,
            "bounds": {**{k: list(v) for k, v in BOUNDS.items()}, "volatility_of_variance": gamma_rule},
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
                "variance": "dV_t = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V",
                "correlation": "d<W^S, W^V>_t = rho dt",
            },
            "initial_conditions": {"spot": "S_0 = spot", "variance": "V_0 = initial_variance"},
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
