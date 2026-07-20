"""Generate heston_02__first_100__american_puts_01__first_100__python_gpu_pytorch_01."""

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

from tools.registry.result.heston.american_puts import (
    generate_validation_reprice_result,
)


def main() -> None:
    print(generate_validation_reprice_result(engine="python_gpu_pytorch", device="cuda"))


if __name__ == "__main__":
    main()
