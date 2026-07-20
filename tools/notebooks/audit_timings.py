"""Audit validation timing metadata and expected engine hierarchy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ENGINE_KEYS = ("cpp_gpu", "python_gpu", "cpp_cpu", "python_cpu")
ANALYTIC_LAUNCH_BOUND = {
    ("black_76", "caplets"),
    ("black_76", "swaptions"),
    ("cir", "zero_coupon_bonds"),
    ("cir_plus_plus", "zero_coupon_bonds"),
    ("g2_plus_plus", "zero_coupon_bonds"),
    ("hull_white", "zero_coupon_bonds"),
}


def _engine_key(engine: str) -> str | None:
    return next((key for key in ENGINE_KEYS if key in engine), None)


def _workload(timing: dict[str, Any]) -> tuple[Any, ...]:
    return (
        timing.get("benchmark_row_count"),
        timing.get("benchmark_row_multiplier"),
        timing.get("benchmark_repetitions"),
        timing.get("benchmark_workload"),
    )


def audit(root: Path) -> list[str]:
    warnings: list[str] = []
    specification_dirs = sorted({path.parent for path in root.rglob("*.yaml")})
    for directory in specification_dirs:
        groups: dict[bool, dict[str, dict[str, Any]]] = {}
        for path in directory.glob("*.yaml"):
            specification = yaml.safe_load(path.read_text(encoding="utf-8"))
            engine = str(specification.get("summary", {}).get("engine", ""))
            key = _engine_key(engine)
            if key is not None:
                groups.setdefault("delta_crn" in engine, {})[key] = specification

        for delta, engines in groups.items():
            label = (
                f"{directory.parent.parent.name}/{directory.parent.name}/"
                f"{'delta' if delta else 'price'}"
            )
            if set(engines) != set(ENGINE_KEYS):
                warnings.append(f"{label}: incomplete four-engine timing set")
                continue
            timings = {
                key: specification.get("timing", {})
                for key, specification in engines.items()
            }
            if len({_workload(timing) for timing in timings.values()}) != 1:
                warnings.append(f"{label}: engines use different benchmark workloads")

            wall = {
                key: float(timing.get("benchmark_seconds", timing["wall_seconds"]))
                for key, timing in timings.items()
            }
            cuda_timing = timings["cpp_gpu"]
            kernel = float(cuda_timing.get(
                "benchmark_kernel_seconds", cuda_timing["kernel_seconds"]
            ))
            model_family = directory.parent.parent.name
            product_family = directory.parent.name
            minimum_cuda_speedup = (
                1.0
                if model_family in {"black_scholes", "rough_bergomi"}
                else 2.0
            )
            if kernel * minimum_cuda_speedup > wall["python_gpu"]:
                expectation = (
                    "does not beat PyTorch GPU"
                    if minimum_cuda_speedup == 1.0
                    else "is less than 2x faster than PyTorch GPU"
                )
                warnings.append(
                    f"{label}: CUDA kernel {expectation} "
                    f"({kernel:.6g}s vs {wall['python_gpu']:.6g}s)"
                )
            launch_bound = (model_family, product_family) in ANALYTIC_LAUNCH_BOUND
            if not launch_bound and wall["python_gpu"] * 2.0 >= wall["python_cpu"]:
                warnings.append(
                    f"{label}: PyTorch GPU is too close to PyTorch CPU "
                    f"({wall['python_gpu']:.6g}s vs {wall['python_cpu']:.6g}s)"
                )
            if not launch_bound and wall["cpp_cpu"] >= wall["python_cpu"]:
                warnings.append(
                    f"{label}: C++ CPU does not beat PyTorch CPU "
                    f"({wall['cpp_cpu']:.6g}s vs {wall['python_cpu']:.6g}s)"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("registry/validation/results")
    )
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    warnings = audit(arguments.root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Audited validation timings: {len(warnings)} warning(s).")
    return 1 if arguments.strict and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
