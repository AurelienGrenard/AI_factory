"""Generate the CIR production model database."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.common.philox import philox_uniforms
from tools.registry.model.core_parameter_databases import write_model_database

DATABASE_ID = "cir_01"
ROW_COUNT = 1_000
SEED = 741_200_001


def model_parameters() -> list[dict[str, float]]:
    uniforms = [philox_uniforms(SEED, ROW_COUNT, stream=i) for i in range(4)]
    rows = []
    for i in range(ROW_COUNT):
        kappa = 0.1 + 1.9 * float(uniforms[0][i])
        theta = 0.005 + 0.115 * float(uniforms[1][i])
        initial_rate = 0.001 + 0.119 * float(uniforms[2][i])
        lower = max(math.sqrt(kappa * theta / 5.0), 0.01)
        upper = min(math.sqrt(12.0 * kappa * theta), 0.30)
        volatility = lower + (upper - lower) * float(uniforms[3][i])
        rows.append({
            "initial_rate": initial_rate,
            "kappa": kappa,
            "theta": theta,
            "volatility": volatility,
        })
    return rows


def main() -> None:
    print(write_model_database(
        database_id=DATABASE_ID,
        model_family="cir",
        parameters=model_parameters(),
        title="CIR production model parameter database",
        construction={
            "method": "conditional random sample",
            "rule": "Kappa, theta, and r0 are uniform; volatility is conditionally uniform under the documented Feller-ratio bounds.",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "bounds": {
                "kappa": [0.1, 2.0], "theta": [0.005, 0.12],
                "initial_rate": [0.001, 0.12],
                "volatility": ["max(sqrt(kappa theta / 5), 0.01)", "min(sqrt(12 kappa theta), 0.30)"],
                "feller_ratio": ["1/6", 10.0],
            },
        },
        parameter_docs={
            "initial_rate": "Initial short rate r0.",
            "kappa": "Positive mean-reversion speed kappa.",
            "theta": "Positive long-run short-rate level theta.",
            "volatility": "Square-root short-rate volatility sigma.",
        },
        dynamics={
            "measure": "Risk-neutral",
            "equations": {"short_rate": "dr_t = kappa (theta - r_t) dt + sigma sqrt(r_t) dW_t"},
            "initial_conditions": {"short_rate": "r_0 = initial_rate"},
        },
        registry_tier="production",
    ))


if __name__ == "__main__":
    main()
