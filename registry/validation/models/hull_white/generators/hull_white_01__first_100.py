"""Slice the first 100 Hull-White production models."""

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

from tools.registry.model.slicing import DEFAULT_VALIDATION_ROW_COUNT, write_model_slice


def main() -> None:
    print(
        write_model_slice(
            project_root=PROJECT_ROOT,
            source_id="hull_white_01",
            target_id="hull_white_01__first_100",
            row_count=DEFAULT_VALIDATION_ROW_COUNT,
        )
    )


if __name__ == "__main__":
    main()
