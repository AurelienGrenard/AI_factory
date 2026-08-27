#!/usr/bin/env python3
"""Enforce the offline CUDA-runner and tools-ownership boundaries."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "tools" / "codegen" / "pricing_bindings"
sys.path.insert(0, str(CODEGEN))

from manifest import MODEL_RECIPE_SPECS, PRICE_VARIANTS  # noqa: E402
RAW_CUDA = (
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaEventCreate",
    "cudaEventDestroy",
)

# These are algorithmically atypical pipelines, not ordinary catalog recipes:
# Longstaff--Schwartz owns extra regression state and a backward execution
# graph. New entries are intentionally not accepted by this checker without
# changing this reviewed list.
EXPLICIT_ESCAPE_HATCHES = {
    f"catalog/model/equity/{model}/prices/american_{side}s/"
    f"{model}_01__american_{side}s_01__01/generator.cpp"
    for model in (
        "bates", "heston", "normal_inverse_gaussian", "variance_gamma"
    )
    for side in ("call", "put")
}

GENERATED_EQUITY_RECIPES = {
    "catalog/model/equity/"
    f"{model.name}/prices/{variant.name}/"
    f"{model.name}_01__{variant.name}_01__01/generator.cpp"
    for model in MODEL_RECIPE_SPECS
    for variant in PRICE_VARIANTS
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    failures: list[str] = []
    generators = sorted((ROOT / "catalog").rglob("generator.cpp"))
    generator_paths = {relative(path) for path in generators}
    missing_generated = GENERATED_EQUITY_RECIPES - generator_paths
    failures.extend(
        f"missing generated equity recipe: {path}"
        for path in sorted(missing_generated)
    )
    for path_text in sorted(GENERATED_EQUITY_RECIPES & generator_paths):
        source = (ROOT / path_text).read_text()
        if not source.startswith("// Generated "):
            failures.append(
                f"equity recipe is not codegen-owned: {path_text}"
            )
        if '"tools/pricing/equity_price_generation.cuh"' not in source:
            failures.append(
                f"generated equity recipe bypasses its orchestrator: {path_text}"
            )
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
            and "offline::pricing::generate_" not in source
            and "pricing::generate_" not in source
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
        f"{len(GENERATED_EQUITY_RECIPES)} generated equity recipes; "
        f"{len(raw)} reviewed algorithmic escape hatches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
