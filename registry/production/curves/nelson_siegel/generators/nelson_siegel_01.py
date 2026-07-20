"""Generate 1,000 admissible Nelson-Siegel production curves."""

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
from tools.registry.curve.core_parameter_databases import write_curve_database

DATABASE_ID = "nelson_siegel_01"
ROW_COUNT = 1_000
CANDIDATE_COUNT = 100_000
SEED = 740_100_001
BOUNDS = {
    "beta0": (0.01, 0.06),
    "short_rate": (-0.005, 0.06),
    "beta2": (-0.04, 0.04),
    "tau": (0.25, 5.0),
}
ZERO_RATE_RANGE = (-0.01, 0.08)
FORWARD_RATE_RANGE = (-0.02, 0.10)


def _uniform(stream: int, lower: float, upper: float) -> np.ndarray:
    values = philox_uniforms(SEED, CANDIDATE_COUNT, stream=stream)
    return lower + (upper - lower) * values


def curve_parameters() -> list[dict[str, float]]:
    beta0 = _uniform(0, *BOUNDS["beta0"])
    short_rate = _uniform(1, *BOUNDS["short_rate"])
    beta1 = short_rate - beta0
    beta2 = _uniform(2, *BOUNDS["beta2"])
    tau = _uniform(3, *BOUNDS["tau"])

    maturities = np.linspace(1.0 / 52.0, 30.0, 256)
    x = maturities[None, :] / tau[:, None]
    exponential = np.exp(-x)
    loading = -np.expm1(-x) / x
    zero_rates = (
        beta0[:, None]
        + beta1[:, None] * loading
        + beta2[:, None] * (loading - exponential)
    )
    forwards = (
        beta0[:, None]
        + beta1[:, None] * exponential
        + beta2[:, None] * x * exponential
    )
    accepted = (
        (zero_rates.min(axis=1) >= ZERO_RATE_RANGE[0])
        & (zero_rates.max(axis=1) <= ZERO_RATE_RANGE[1])
        & (forwards.min(axis=1) >= FORWARD_RATE_RANGE[0])
        & (forwards.max(axis=1) <= FORWARD_RATE_RANGE[1])
    )
    indices = np.flatnonzero(accepted)[:ROW_COUNT]
    if indices.size != ROW_COUNT:
        raise RuntimeError(
            f"Only {indices.size} admissible curves among {CANDIDATE_COUNT} candidates."
        )
    return [
        {
            "beta0": round(float(beta0[index]), 12),
            "beta1": round(float(beta1[index]), 12),
            "beta2": round(float(beta2[index]), 12),
            "tau": round(float(tau[index]), 12),
        }
        for index in indices
    ]


def main() -> None:
    path = write_curve_database(
        database_id=DATABASE_ID,
        curve_family="nelson_siegel",
        parameters=curve_parameters(),
        title="Nelson Siegel production zero-coupon curve database",
        parameter_docs={
            "beta0": "Long-rate level.",
            "beta1": "Short-rate slope loading.",
            "beta2": "Curvature loading.",
            "tau": "Positive decay time in years.",
        },
        equations={
            "zero_rate": "y(0,T) = beta0 + beta1 A(T/tau) + beta2 (A(T/tau) - exp(-T/tau))",
            "loading": "A(x) = (1 - exp(-x)) / x",
            "discount_factor": "P(0,T) = exp(-T y(0,T))",
            "instantaneous_forward": "f(0,T) = beta0 + beta1 exp(-T/tau) + beta2 (T/tau) exp(-T/tau)",
        },
        construction={
            "method": "uniform rejection sample",
            "rule": "Accept the first 1,000 candidates whose zero and instantaneous-forward curves stay inside the admissibility ranges on 256 maturities from 1/52 to 30 years.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "candidate_count": CANDIDATE_COUNT,
            "bounds": {key: list(value) for key, value in BOUNDS.items()},
            "admissibility": {
                "zero_rate": list(ZERO_RATE_RANGE),
                "instantaneous_forward": list(FORWARD_RATE_RANGE),
            },
        },
        registry_tier="production",
    )
    print(path)


if __name__ == "__main__":
    main()
