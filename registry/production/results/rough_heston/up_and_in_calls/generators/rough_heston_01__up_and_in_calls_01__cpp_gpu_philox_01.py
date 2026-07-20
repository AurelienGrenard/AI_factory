"""Generate the production CUDA barrier result."""
import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "registry").is_dir() and (parent / "src_cpp").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry.result.rough_heston.up_and_in_calls import generate_production_cpp_gpu_result

if __name__ == "__main__":
    print(generate_production_cpp_gpu_result())
