"""Slice up_and_in_calls_01 for validation."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.product.slicing import write_product_slice
if __name__ == "__main__":
    print(write_product_slice(project_root=PROJECT_ROOT, source_id="up_and_in_calls_01", target_id="up_and_in_calls_01__first_100", row_count=100))
