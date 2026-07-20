"""Generate Rough Bergomi autocall production prices."""
import sys
from pathlib import Path
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
from tools.registry.result.rough_bergomi.autocalls import generate_production_cpp_gpu_result
if __name__ == "__main__":
    print(generate_production_cpp_gpu_result())
