"""Reprice the production barrier audit slice with python_cpu_pytorch."""
import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.result.rough_heston.up_and_out_calls import generate_validation_reprice_result

if __name__ == "__main__":
    print(generate_validation_reprice_result(engine="python_cpu_pytorch", device="cpu"))
