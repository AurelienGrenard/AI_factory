"""Generate Rough Bergomi parameters with variable rates for autocalls."""

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

DATABASE_ID = "rough_bergomi_02"
ROW_COUNT = 1_000
SEED = 720_000_101
BOUNDS = {
    "hurst": (0.01, 0.40),
    "eta": (1.0, 3.0),
    "correlation": (-0.95, -0.30),
    "forward_variance": (0.01, 0.12),
    "risk_free_rate": (0.0, 0.08),
    "dividend_yield": (0.0, 0.05),
}


def model_parameters() -> list[dict[str, float]]:
    columns = philox_uniform_columns(seed=SEED, row_count=ROW_COUNT, bounds=BOUNDS)
    return [{
        "spot": 1.0,
        "risk_free_rate": float(columns["risk_free_rate"][i]),
        "dividend_yield": float(columns["dividend_yield"][i]),
        "forward_variance": float(columns["forward_variance"][i]),
        "eta": float(columns["eta"][i]),
        "hurst": float(columns["hurst"][i]),
        "correlation": float(columns["correlation"][i]),
    } for i in range(ROW_COUNT)]


def main() -> None:
    print(write_model_database(
        database_id=DATABASE_ID,
        model_family="rough_bergomi",
        parameters=model_parameters(),
        title=f"Rough Bergomi production model parameter database {DATABASE_ID}",
        construction={
            "method": "random sample",
            "rule": "All model parameters are sampled independently and uniformly.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "spot": 1.0,
            "row_count": ROW_COUNT,
            "bounds": {key: list(value) for key, value in BOUNDS.items()},
        },
        parameter_docs={
            "spot": "Initial spot.",
            "risk_free_rate": "Continuously compounded risk-free rate.",
            "dividend_yield": "Continuously compounded dividend yield.",
            "forward_variance": "Flat initial forward variance xi_0.",
            "eta": "Rough Bergomi volatility of volatility eta.",
            "hurst": "Hurst exponent H.",
            "correlation": "Spot/volatility Brownian correlation rho.",
        },
        dynamics={
            "measure": "Risk-neutral",
            "equations": {
                "spot": "dS_t / S_t = (r - q) dt + sqrt(V_t) dZ_t",
                "variance": "V_t = xi_0(t) exp(eta W_tilde_t^H - 0.5 eta^2 t^(2H))",
                "volterra_process": "W_tilde_t^H = sqrt(2H) integral_0^t (t - s)^(H - 0.5) dW_s",
                "correlation": "d<Z, W>_t = rho dt",
            },
            "initial_forward_variance": {"convention": "xi_0(t) is flat", "value": "xi_0(t) = forward_variance"},
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
