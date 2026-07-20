"""Generate black_scholes_01__european_calls_01__cpp_gpu_philox_01."""

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

from tools.registry.result.black_scholes.european_calls import PRODUCTION_ROW_COUNT, generate_result

if __name__ == "__main__":
    print(generate_result(tier="production", model_id="black_scholes_01", product_id="european_calls_01", result_version="01", engine="cpp_gpu_philox", row_count=PRODUCTION_ROW_COUNT))

