#!/usr/bin/env python3
"""Enforce the offline CUDA-runner and tools-ownership boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_CUDA = (
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaEventCreate",
    "cudaEventDestroy",
)

# These are algorithmically atypical pipelines, not ordinary catalog recipes:
# Longstaff--Schwartz owns extra regression state, while the legacy FFT recipe
# owns a model-specific cuFFTDx workspace. New entries are intentionally not
# accepted by this checker without changing this reviewed list.
EXPLICIT_ESCAPE_HATCHES = {
    f"catalog/model/equity/{model}/prices/american_{side}s/"
    f"{model}_01__american_{side}s_01__01/generator.cpp"
    for model in (
        "bates", "heston", "normal_inverse_gaussian", "variance_gamma"
    )
    for side in ("call", "put")
} | {
    "catalog/model/equity/rough_bergomi/prices/european_calls/"
    "rough_bergomi_01__european_calls_01__01/generator.cpp",
    "catalog/model/equity/rough_bergomi/prices/european_puts/"
    "rough_bergomi_01__european_puts_01__01/generator.cpp",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    failures: list[str] = []
    generators = sorted((ROOT / "catalog").rglob("generator.cpp"))
    raw = {
        relative(path)
        for path in generators
        if any(token in path.read_text() for token in RAW_CUDA)
    }
    unexpected = raw - EXPLICIT_ESCAPE_HATCHES
    stale = EXPLICIT_ESCAPE_HATCHES - raw
    failures.extend(
        f"manual CUDA outside reviewed escape hatch: {path}"
        for path in sorted(unexpected)
    )
    failures.extend(
        f"stale CUDA escape hatch (remove it): {path}"
        for path in sorted(stale)
    )

    for path in generators:
        path_text = relative(path)
        if "/prices/" not in path_text or path_text in raw:
            continue
        source = path.read_text()
        if (
            "offline::cuda::run_" not in source
            and '"tools/pricing/' not in source
        ):
            failures.append(
                f"price recipe bypasses the shared runner: {path_text}"
            )

    forbidden = [
        ROOT / "tools/datasets/dataset.hpp",
        ROOT / "tools/datasets/dataset.cpp",
        ROOT / "tools/datasets/european_swaption_price_generation.hpp",
        ROOT / "tools/datasets/bermudan_swaption_price_generation.hpp",
    ]
    failures.extend(
        f"retired tools monolith/helper still exists: {relative(path)}"
        for path in forbidden
        if path.exists()
    )

    for path in sorted((ROOT / "tools/pricing").glob("*")):
        if path.is_file() and any(
            token in path.read_text() for token in RAW_CUDA
        ):
            failures.append(
                f"pricing helper manually owns CUDA resources: {relative(path)}"
            )

    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"{len(generators)} catalog recipes checked; "
        f"{len(raw)} reviewed algorithmic escape hatches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
