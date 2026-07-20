"""Generate down_and_out_calls_01."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.product.barrier_calls import write_barrier_call_database
if __name__ == "__main__":
    write_barrier_call_database("down", "out")
