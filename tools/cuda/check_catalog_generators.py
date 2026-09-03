#!/usr/bin/env python3
"""Enforce the offline CUDA-runner and tools-ownership boundaries."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "tools" / "codegen" / "pricing_bindings"
sys.path.insert(0, str(CODEGEN))

from manifest import (  # noqa: E402
    AMERICAN_RECIPE_SPECS,
    MODEL_RECIPE_SPECS,
    PRICE_VARIANTS,
)
from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    DEFERRED_DATASET_SPECS,
    ENGINE_SPECS,
    MODEL_SPECS,
    PRODUCT_SPECS,
    resolve_price_capability,
)
RAW_CUDA = (
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaEventCreate",
    "cudaEventDestroy",
)

GENERATED_AMERICAN_RECIPES = {
    f"catalog/model/equity/{model}/prices/american_{side}s/"
    f"{model}_01__american_{side}s_01__01/generator.cpp"
    for model in (spec.model for spec in AMERICAN_RECIPE_SPECS)
    for side in ("call", "put")
}

GENERATED_EQUITY_RECIPES = {
    "catalog/model/equity/"
    f"{model.name}/prices/{variant.name}/"
    f"{model.name}_01__{variant.name}_01__01/generator.cpp"
    for model in MODEL_RECIPE_SPECS
    for variant in PRICE_VARIANTS
} | GENERATED_AMERICAN_RECIPES


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    failures: list[str] = []
    generators = sorted((ROOT / "catalog").rglob("generator.cpp"))
    generator_paths = {relative(path) for path in generators}
    expected_recipe_paths = {
        dataset.recipe_path for dataset in AVAILABLE_DATASET_SPECS
    }
    deferred_recipe_paths = {
        dataset.recipe_path for dataset in DEFERRED_DATASET_SPECS
    }
    failures.extend(
        f"undeclared catalog recipe: {path}"
        for path in sorted(generator_paths - expected_recipe_paths)
    )
    failures.extend(
        f"missing declared catalog recipe: {path}"
        for path in sorted(expected_recipe_paths - generator_paths)
    )
    failures.extend(
        f"deferred recipe exists without an available capability: {path}"
        for path in sorted(generator_paths & deferred_recipe_paths)
    )

    if len({engine.name for engine in ENGINE_SPECS}) != len(ENGINE_SPECS):
        failures.append("duplicate EngineSpec name")
    if len({model.name for model in MODEL_SPECS}) != len(MODEL_SPECS):
        failures.append("duplicate ModelSpec name")
    if len({
        (product.asset_class, product.name) for product in PRODUCT_SPECS
    }) != len(PRODUCT_SPECS):
        failures.append("duplicate ProductSpec key")
    if len(expected_recipe_paths) != len(AVAILABLE_DATASET_SPECS):
        failures.append("duplicate available DatasetSpec recipe path")

    for dataset in AVAILABLE_DATASET_SPECS:
        if dataset.dataset_kind != "prices":
            continue
        try:
            resolved = resolve_price_capability(
                dataset.model or "",
                dataset.product or "",
                dataset.variant or "",
                dataset.curve,
            )
        except KeyError as error:
            failures.append(str(error))
            continue
        if resolved != dataset:
            failures.append(
                f"capability resolver returned the wrong DatasetSpec: "
                f"{dataset.recipe_path}"
            )

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
        expected_helper = (
            '"tools/pricing/american_option_price_generation.cuh"'
            if path_text in GENERATED_AMERICAN_RECIPES
            else '"tools/pricing/equity_price_generation.cuh"'
        )
        if expected_helper not in source:
            failures.append(
                f"generated equity recipe bypasses its orchestrator: {path_text}"
            )
    raw = {
        relative(path)
        for path in generators
        if any(token in path.read_text() for token in RAW_CUDA)
    }
    failures.extend(
        f"catalog recipe manually owns CUDA resources: {path}"
        for path in sorted(raw)
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
        if (
            "BatchedMonteCarloProfile profile" in source
            and "BatchedMonteCarloProfile profile{\n        1'048'576U,"
                not in source
        ):
            failures.append(
                f"Monte Carlo price recipe does not use 2^20 paths: "
                f"{path_text}"
            )
        if (
            "AmericanOptionProfile profile" in source
            and "AmericanOptionProfile profile{\n        1U << 20U,"
                not in source
        ):
            failures.append(
                f"American price recipe does not use 2^20 paths: {path_text}"
            )
        if (
            "bermudan_swaption" in path_text
            and "constexpr std::size_t paths = 1U << 20U;" not in source
        ):
            failures.append(
                f"Bermudan price recipe does not use 2^20 paths: {path_text}"
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

    expected_sample_units = {
        (
            ROOT / "src" / "model" / model.asset_class
            / ("rough" if model.family.startswith("rough_") else "markovian")
            / model.name / "sample.cu"
            if model.asset_class == "equity"
            else ROOT / "src" / "model" / "fixed_income" / model.name
            / "sample.cu"
        )
        for model in MODEL_SPECS
        if model.sample_binding_status == "available"
    }
    actual_sample_units = set((ROOT / "src" / "model").rglob("sample.cu"))
    failures.extend(
        f"missing declared sample binding: {relative(path)}"
        for path in sorted(expected_sample_units - actual_sample_units)
    )
    failures.extend(
        f"undeclared sample binding: {relative(path)}"
        for path in sorted(actual_sample_units - expected_sample_units)
    )

    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"{len(generators)} catalog recipes checked; "
        f"{len(GENERATED_EQUITY_RECIPES)} generated equity recipes; "
        f"{len(DEFERRED_DATASET_SPECS)} explicit deferred sample recipes; "
        "no raw-CUDA recipe escape hatch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
