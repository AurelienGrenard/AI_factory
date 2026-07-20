"""Generate rough_heston_01__lookback_fixed_calls_01__cpp_gpu_philox_delta_crn_01."""

from __future__ import annotations

import argparse
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

from tools.registry.result.rough_heston.lookback_fixed_calls import (
    DEFAULT_RELATIVE_BUMP,
    DEFAULT_NUM_PATHS,
    DEFAULT_TARGET_DT,
    PRODUCTION_ROW_COUNT,
    generate_production_cpp_gpu_delta_crn_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-count", type=int, default=PRODUCTION_ROW_COUNT)
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--target-dt", default=DEFAULT_TARGET_DT)
    parser.add_argument("--relative-bump", type=float, default=DEFAULT_RELATIVE_BUMP)
    args = parser.parse_args()
    print(
        generate_production_cpp_gpu_delta_crn_result(
            row_count=args.row_count,
            num_paths=args.num_paths,
            target_dt=args.target_dt,
            relative_bump=args.relative_bump,
        )
    )


if __name__ == "__main__":
    main()
