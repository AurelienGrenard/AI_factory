"""Generate the european_calls_01 production product database."""

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

from tools.registry.product.recipe_helpers import write_product_recipe_database
from tools.registry.product.terminal_calls import (
    LOG_MONEYNESS_SLOPE,
    MAX_MATURITY,
    MIN_MATURITY,
    ROW_COUNT,
    SEED,
    terminal_call_parameters,
)

DATABASE_ID = "european_calls_01"
def main() -> None:
    write_product_recipe_database(
        database_id=DATABASE_ID,
        product_family="european_calls",
        title="European call production product parameter database",
        payoff_key="european_call",
        expression="max(S_T - K, 0)",
        scaling_rule="V(s, K) = s * V(1, K / s)",
        parameters=terminal_call_parameters(),
        construction={
            "method": "random sample",
            "rule": "T is uniform on [1/12, 3]. Conditional on T, log(K) is uniform on [-aT, aT].",
            "rng": "Project Philox-4x32-10 generator",
            "seed": SEED,
            "row_count": ROW_COUNT,
            "bounds": {
                "maturity": ["1/12", 3.0],
                "log_strike_given_maturity": "uniform on [-aT, aT]",
                "a": LOG_MONEYNESS_SLOPE,
            },
        },
        parameter_docs={
            "strike": "Strike in normalized spot units.",
            "maturity": "Maturity in years.",
        },
        registry_tier="production",
    )


if __name__ == "__main__":
    main()
