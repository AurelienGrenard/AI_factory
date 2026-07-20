"""Slice the first 100 rows of heston_03."""
import sys
from pathlib import Path
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
from tools.registry.model.slicing import DEFAULT_VALIDATION_ROW_COUNT, write_model_slice
if __name__ == "__main__":
    print(write_model_slice(project_root=PROJECT_ROOT, source_id="heston_03", target_id="heston_03__first_100", row_count=DEFAULT_VALIDATION_ROW_COUNT))
