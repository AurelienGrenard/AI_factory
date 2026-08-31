#!/usr/bin/env python3
"""Enforce the offline CUDA-runner and tools-ownership boundaries."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
CODEGEN = ROOT / "tools" / "codegen" / "pricing_bindings"
sys.path.insert(0, str(CODEGEN))

from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    DEFERRED_DATASET_SPECS,
    ENGINE_SPECS,
    MODEL_SPECS,
    PRODUCT_SPECS,
    RNG_DOMAIN_SPECS,
    resolve_rng_domain,
    resolve_price_capability,
    validate_rng_domain_specs,
)
RAW_CUDA = (
    "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaEventCreate",
    "cudaEventDestroy",
)

GENERATED_RECIPES = {
    dataset.recipe_path
    for dataset in AVAILABLE_DATASET_SPECS
    if dataset.owner == "generated"
}
GENERATED_AMERICAN_RECIPES = {
    dataset.recipe_path
    for dataset in AVAILABLE_DATASET_SPECS
    if dataset.engine in {"equity_lsm_fixed", "equity_lsm_exact"}
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    failures: list[str] = []
    try:
        validate_rng_domain_specs(RNG_DOMAIN_SPECS)
    except ValueError as error:
        failures.append(str(error))
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

    missing_generated = GENERATED_RECIPES - generator_paths
    failures.extend(
        f"missing generated recipe: {path}"
        for path in sorted(missing_generated)
    )
    for path_text in sorted(GENERATED_RECIPES & generator_paths):
        source = (ROOT / path_text).read_text()
        if not source.startswith("// Generated "):
            failures.append(
                f"generated recipe is not codegen-owned: {path_text}"
            )
        if "/prices/" not in path_text:
            continue
        expected_helpers = (
            ('"tools/pricing/american_option_price_generation.cuh"',)
            if path_text in GENERATED_AMERICAN_RECIPES
            else ('"tools/pricing/equity_price_generation.cuh"',)
            if "/model/equity/" in path_text
            else (
                '"tools/cuda/pricing_runner.cuh"',
                '"tools/pricing/european_swaption_price_generation.cuh"',
            )
        )
        if not any(helper in source for helper in expected_helpers):
            failures.append(
                f"generated recipe bypasses its orchestrator: {path_text}"
            )

    for dataset in AVAILABLE_DATASET_SPECS:
        path = ROOT / dataset.recipe_path
        if not path.is_file():
            continue
        source = path.read_text()
        try:
            rng_domain = resolve_rng_domain(dataset)
        except KeyError:
            rng_domain = None
        if rng_domain is not None:
            for stream in rng_domain.streams:
                seed_literal = f'{rng_domain.seed(stream)}ULL'
                if source.count(seed_literal) != 1:
                    failures.append(
                        f"recipe does not use exactly one declared {stream} "
                        f"RNG seed {seed_literal}: {dataset.recipe_path}"
                    )
        literals = "".join(re.findall(
            r'"([^"\\]*(?:\\.[^"\\]*)*)"', source
        ))
        if dataset.dataset_kind == "samples":
            helper = (
                ROOT / "tools" / "sampling" / "generated"
                / f"{dataset.model}_sample_generation.cuh"
            )
            helper_literals = "".join(re.findall(
                r'"([^"\\]*(?:\\.[^"\\]*)*)"', helper.read_text()
            ))
            prefix = f"{dataset.source_prefix}/samples/"
            for label, value in (
                ("dataset", f"datasets/{prefix}"),
                ("catalog", f"catalog/{prefix}"),
                (
                    "URL",
                    "https://datasets.ai-factory.example/v1/" + prefix,
                ),
            ):
                if value not in helper_literals:
                    failures.append(
                        f"sample helper has a noncanonical {label} prefix: "
                        f"{relative(helper)}"
                    )
        elif dataset.dataset_kind == "prices" and dataset.owner == "generated":
            for label, value in (
                ("dataset", dataset.dataset_path),
                ("catalog", dataset.catalog_yaml_path),
                ("URL", dataset.url),
            ):
                if value not in literals:
                    failures.append(
                        f"generated recipe has the wrong {label} mapping: "
                        f"{dataset.recipe_path}"
                    )
        elif dataset.dataset_kind == "prices":
            if (
                dataset.engine == "fixed_income_lsm"
                and '"product/bermudan_swaption/dataset.hpp"' not in source
            ):
                failures.append(
                    "handwritten Bermudan recipe does not directly declare "
                    f"its product loader: {dataset.recipe_path}"
                )
            if f'"{dataset.model}"' not in source:
                failures.append(
                    f"handwritten price recipe has the wrong model mapping: "
                    f"{dataset.recipe_path}"
                )
            if dataset.curve is not None and f'"{dataset.curve}"' not in source:
                failures.append(
                    f"handwritten price recipe has the wrong curve mapping: "
                    f"{dataset.recipe_path}"
                )
            side = (
                "payer" if "payer" in (dataset.variant or "") else "receiver"
            )
            if f'"{side}"' not in source:
                failures.append(
                    f"handwritten price recipe has the wrong side mapping: "
                    f"{dataset.recipe_path}"
                )
        else:
            for label, value in (
                ("dataset", dataset.dataset_path),
                ("catalog", dataset.catalog_yaml_path),
            ):
                if value not in literals:
                    failures.append(
                        f"parameter recipe has the wrong {label} mapping: "
                        f"{dataset.recipe_path}"
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
        f"{len(GENERATED_RECIPES)} generated recipes; "
        f"{len(DEFERRED_DATASET_SPECS)} explicit deferred sample recipes; "
        "no raw-CUDA recipe escape hatch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
