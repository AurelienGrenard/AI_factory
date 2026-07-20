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

from tools.registry.result.black_scholes.american_puts import AUDIT_ROW_COUNT, generate_result


def main() -> None:
    print(
        generate_result(
            tier="validation",
            model_id="black_scholes_01__first_100",
            product_id="american_puts_01__first_100",
            result_version="01",
            engine="python_cpu_pytorch",
            row_count=AUDIT_ROW_COUNT,
        )
    )


if __name__ == "__main__":
    main()
