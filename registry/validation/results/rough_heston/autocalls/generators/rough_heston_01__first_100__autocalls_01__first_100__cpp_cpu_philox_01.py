"""Reprice the autocall validation slice with cpp_cpu_philox."""
import sys
from pathlib import Path
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "registry").is_dir() and (p / "src_cpp").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))
from tools.registry.result.rough_heston.autocalls import generate_validation_reprice_result
if __name__ == "__main__":
    print(generate_validation_reprice_result(engine="cpp_cpu_philox", device="cpu"))
