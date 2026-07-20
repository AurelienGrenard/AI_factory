"""Generate the first production rough Heston parameter database."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns, uniform_between

DATABASE_ID = "rough_heston_01"
ROW_COUNT = 1_000
SEED = 725_000_101
BOUNDS = {
    "kappa": (0.5, 4.0),
    "theta": (0.01, 0.15),
    "rho": (-0.95, -0.30),
    "initial_variance": (0.01, 0.12),
    "hurst": (0.05, 0.40),
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
    return [
        {
            "spot": 1.0,
            "risk_free_rate": float(columns["risk_free_rate"][index]),
            "dividend_yield": float(columns["dividend_yield"][index]),
            "initial_variance": float(columns["initial_variance"][index]),
            "theta": float(theta[index]),
            "kappa": float(kappa[index]),
            "volatility_of_variance": float(gamma[index]),
            "hurst": float(columns["hurst"][index]),
            "rho": float(columns["rho"][index]),
        }
        for index in range(ROW_COUNT)
    ]


def main() -> None:
    gamma_rule = (
        "gamma is uniform on [max(sqrt(kappa * theta / 5), 0.1), "
        "min(sqrt(12 * kappa * theta), 0.8)]."
    )
    print(
        write_model_database(
            database_id=DATABASE_ID,
            model_family="rough_heston",
            parameters=model_parameters(),
            title=f"Rough Heston production model parameter database {DATABASE_ID}",
            construction={
                "method": "random sample",
                "rule": "All listed parameters are sampled independently and uniformly; "
                + gamma_rule,
                "rng": "Project Philox-4x32-10 generator",
                "seed": SEED,
                "spot": 1.0,
                "row_count": ROW_COUNT,
                "bounds": {
                    **{key: list(value) for key, value in BOUNDS.items()},
                    "volatility_of_variance": gamma_rule,
                },
            },
            parameter_docs={
                "spot": "Initial spot.",
                "risk_free_rate": "Continuously compounded risk-free rate.",
                "dividend_yield": "Continuously compounded dividend yield.",
                "initial_variance": "Initial variance v0.",
                "theta": "Long-run variance theta.",
                "kappa": "Volterra variance mean-reversion speed kappa.",
                "volatility_of_variance": "Volatility of variance gamma.",
                "hurst": "Hurst exponent H of the fractional kernel.",
                "rho": "Spot/variance Brownian correlation rho.",
            },
            dynamics={
                "measure": "Risk-neutral",
                "equations": {
                    "spot": "dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S",
                    "variance": "V_t = v0 + integral_0^t K(t-s) [kappa(theta-V_s) ds + gamma sqrt(V_s) dW_s^V]",
                    "kernel": "K(t) = t^(H-1/2) / Gamma(H+1/2)",
                    "correlation": "d<W^S, W^V>_t = rho dt",
                },
                "initial_conditions": {
                    "spot": "S_0 = spot",
                    "variance": "V_0 = initial_variance",
                },
                "numerical_representation": {
                    "type": "eight-factor positive-exponential approximation of K",
                    "quadrature": "geometric Laplace-measure partition with interval mass and first-moment nodes",
                    "variance_step": "explicit lifted Euler with full truncation",
                },
            },
            registry_tier="production",
        )
    )


if __name__ == "__main__":
    main()
