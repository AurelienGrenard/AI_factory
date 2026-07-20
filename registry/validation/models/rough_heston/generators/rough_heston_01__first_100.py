"""Slice the first 100 rows of rough_heston_01."""

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.model.slicing import DEFAULT_VALIDATION_ROW_COUNT, write_model_slice

if __name__ == "__main__":
    print(
        write_model_slice(
            project_root=PROJECT_ROOT,
            source_id="rough_heston_01",
            target_id="rough_heston_01__first_100",
            row_count=DEFAULT_VALIDATION_ROW_COUNT,
        )
    )
