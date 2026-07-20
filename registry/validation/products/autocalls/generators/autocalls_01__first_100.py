"""Slice the first 100 rows of autocalls_01."""
import sys
from pathlib import Path
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
from tools.registry.product.slicing import DEFAULT_VALIDATION_ROW_COUNT, write_product_slice
if __name__ == "__main__":
    print(write_product_slice(project_root=PROJECT_ROOT, source_id="autocalls_01", target_id="autocalls_01__first_100", row_count=DEFAULT_VALIDATION_ROW_COUNT))
