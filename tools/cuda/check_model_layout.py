#!/usr/bin/env python3
"""Enforce model/product ownership, file summaries and codegen discoverability."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "model"
TEMPLATE_ROOT = ROOT / "tools" / "codegen" / "pricing_bindings" / "templates"
CODEGEN_ROOT = ROOT / "tools" / "codegen" / "pricing_bindings"
SOURCE_REFERENCE_INDEX = ROOT / "docs" / "model-and-curve-reference-index.md"
sys.path.insert(0, str(CODEGEN_ROOT))

from capability_manifest import (  # noqa: E402
    AVAILABLE_DATASET_SPECS,
    ENGINE_SPECS,
    GENERATED_PRODUCT_BINDING_PATHS,
    GENERATED_PRODUCT_BINDING_SPECS,
)
CPP_SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp"}
TEXT_SUFFIXES = CPP_SUFFIXES | {".cmake", ".md", ".py", ".txt"}
RESPONSIBILITY_SUFFIXES = CPP_SUFFIXES | {".cmake", ".py", ".tpl"}
CURVE_NAMES = {"nelson_siegel", "svensson"}
CANONICAL_INFRASTRUCTURE_FILENAMES = {
    "analytics.cuh",
    "analytics_impl.cuh",
    "dataset.cpp",
    "dataset.hpp",
    "dynamics.cuh",
    "dynamics_impl.cuh",
    "fitted_analytics.cuh",
    "markovian_n_factor_preparation.hpp",
    "markovian_n_factor_pricing.cuh",
    "parameters.hpp",
    "sample.cu",
    "sample.cuh",
    "state.hpp",
    "volterra_fft_pricing.cuh",
    "volterra_fft_workspace.cuh",
}
AMBIGUOUS_MODEL_FILENAMES = {
    "hybrid_pricing.cuh",
    "pricing_workspace.cuh",
    "markovian_pricing.cuh",
    "numerics.hpp",
}
AMBIGUOUS_FLOATING_TIME_COORDINATE = re.compile(
    r"\b(?:float|double)\s+(?:maturity|time|valuation_time|option_expiry|"
    r"bond_maturity|start_time|end_time|start|end)\b"
)
LOCAL_README_DUPLICATED_INVENTORY = re.compile(
    r"^## Files$|<summary>Exact .*signatures</summary>",
    re.MULTILINE,
)
STALE_FAMILY_TAXONOMY = re.compile(
    r"catalog identifiers and dataset paths do not\s+include"
)
SUMMARY_ROLE_TERMS = {
    "adapter",
    "analytics",
    "binding",
    "composition",
    "contract",
    "dataset",
    "declaration",
    "definition",
    "dynamics",
    "implementation",
    "interface",
    "interfaces",
    "loader",
    "launch",
    "mapping",
    "parameter",
    "parameters",
    "preparation",
    "simulation",
    "state",
    "transition",
    "workspace",
}
PUBLIC_HEADER_TERMS = {
    "contract", "declaration", "declarations", "interface", "interfaces",
    "public",
}
IMPLEMENTATION_HEADER_TERMS = {"definition", "device", "implementation"}

REQUIRED_TEMPLATE_PATHS = {
    "pricing/markovian/product_binding.cuh.tpl",
    "pricing/markovian/product_binding.cu.tpl",
    "pricing/rough/markovian_n_factor/product_binding.cuh.tpl",
    "pricing/rough/markovian_n_factor/product_binding.cu.tpl",
    "pricing/rough/volterra_fft/product_binding.cuh.tpl",
    "pricing/rough/volterra_fft/product_binding.cu.tpl",
    "sampling/markovian/model_binding.cuh.tpl",
    "sampling/markovian/model_binding.cu.tpl",
    "sampling/rough/markovian_n_factor/model_binding.cuh.tpl",
    "sampling/rough/markovian_n_factor/model_binding.cu.tpl",
    "sampling/rough/volterra_fft/model_binding.cuh.tpl",
    "sampling/rough/volterra_fft/model_binding.cu.tpl",
} | {
    f"{spec.template_family}/{spec.product}.{suffix}.tpl"
    for spec in GENERATED_PRODUCT_BINDING_SPECS
    if spec.engine in {"equity_closed_form", "fixed_income_closed_form"}
    and spec.template_family is not None
    for suffix in ("cuh", "cu")
} | {
    dataset.template
    for dataset in AVAILABLE_DATASET_SPECS
    if dataset.owner == "generated" and dataset.template is not None
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_generated_implementation(path: Path) -> bool:
    """Classify only outputs owned by the pricing-binding generator."""
    path_text = relative(path)
    if path_text == "cmake/generated/EquityPricingBindings.cmake":
        return True
    if path_text.startswith("tools/sampling/generated/"):
        return True
    if path.name in {"sample.cu", "sample.cuh"} and path_text.startswith(
        "src/model/"
    ):
        return True
    return path_text in GENERATED_PRODUCT_BINDING_PATHS


def responsibility_format(path: Path, source: str) -> tuple[str, int]:
    """Return the declared comment format and permitted preamble length."""
    if path.suffix == ".py":
        has_shebang = source.startswith("#!")
        return "python", 1 if has_shebang else 0
    if path.suffix == ".cmake" or path.name == "CMakeLists.txt":
        return "cmake", 0
    return "cpp", 0


def top_level_responsibility_summary(
    path: Path,
    source: str,
) -> str | None:
    """Extract a handwritten file summary after its explicit format preamble."""
    source_format, preamble_lines = responsibility_format(path, source)
    lines = source.splitlines()[preamble_lines:]
    first_line = next((line.strip() for line in lines if line.strip()), "")
    if source_format == "python":
        for quote in ('"""', "'''"):
            if first_line.startswith(quote):
                summary = first_line[len(quote):]
                if summary.endswith(quote):
                    summary = summary[:-len(quote)]
                return summary.strip()
        return None
    marker = "# " if source_format == "cmake" else "// "
    if not first_line.startswith(marker):
        return None
    return first_line[len(marker):].strip()


def validate_top_level_responsibility(path: Path, source: str) -> str | None:
    summary = top_level_responsibility_summary(path, source)
    if summary is None:
        return "handwritten implementation lacks a top-level responsibility summary"
    if not 20 <= len(summary) <= 180 or not summary.endswith("."):
        return "handwritten responsibility summary must be one short sentence"
    return None


def responsibility_inventory() -> list[Path]:
    paths = [ROOT / "CMakeLists.txt"]
    for directory in ("src", "tools", "tests", "cmake"):
        paths.extend(
            path for path in (ROOT / directory).rglob("*")
            if path.is_file() and path.suffix in RESPONSIBILITY_SUFFIXES
        )
    return sorted(paths)


def model_product_boundary_index(path: Path) -> int:
    parts = path.relative_to(MODEL_ROOT).parts
    if parts[0] == "equity":
        return 3
    if parts[0] == "fixed_income":
        return 2
    raise ValueError(f"unknown model domain: {relative(path)}")


def validate_model_path(path: Path, under_product: bool) -> list[str]:
    """Validate every level below the model, not just the file basename."""
    parts = path.relative_to(MODEL_ROOT).parts
    boundary = model_product_boundary_index(path)
    tail = parts[boundary:]
    failures: list[str] = []
    if under_product:
        allowed_lengths = {2} if parts[0] == "equity" else {2, 3}
        if len(tail) not in allowed_lengths:
            failures.append(f"unexpected product nesting: {relative(path)}")
        if len(tail) == 3 and tail[1] not in CURVE_NAMES:
            failures.append(
                f"unrecognized fixed-income product qualifier: {relative(path)}"
            )
    else:
        if len(tail) > 1:
            is_curve_analytics = (
                parts[0] == "fixed_income"
                and len(tail) == 2
                and tail[0] in CURVE_NAMES
                and tail[1] in {"analytics.cuh", "analytics_impl.cuh"}
            )
            if not is_curve_analytics:
                failures.append(
                    f"unexpected model-infrastructure nesting: {relative(path)}"
                )
    return failures


def main() -> int:
    failures: list[str] = []
    pricing_composition = (
        ROOT / "docs" / "cuda" / "pricing-policy-composition.md"
    ).read_text()
    for engine in ENGINE_SPECS:
        if f"`{engine.name}`" not in pricing_composition:
            failures.append(
                "pricing composition omits active engine: "
                f"{engine.name}"
            )
    for stale_claim in (
        "seront ajoutees apres validation",
        "presente encore deux facades equivalentes",
    ):
        if stale_claim in pricing_composition:
            failures.append(
                f"pricing composition contains a stale claim: {stale_claim}"
            )
    source_readmes = sorted((ROOT / "src").rglob("README.md"))
    source_reference_text = SOURCE_REFERENCE_INDEX.read_text()
    indexed_source_readmes = [
        (SOURCE_REFERENCE_INDEX.parent / target).resolve()
        for target in re.findall(
            r"\((\.\./src/[^)#]+/README\.md)\)",
            source_reference_text,
        )
    ]
    indexed_source_readme_set = set(indexed_source_readmes)
    source_readme_set = {path.resolve() for path in source_readmes}
    for path in sorted(source_readme_set - indexed_source_readme_set):
        failures.append(f"source README is not indexed: {relative(path)}")
    for path in sorted(indexed_source_readme_set - source_readme_set):
        failures.append(
            "source README index contains a stale link: "
            f"{path.relative_to(ROOT).as_posix()}"
        )
    if len(indexed_source_readmes) != len(indexed_source_readme_set):
        failures.append("source README index contains a duplicate link")
    for path in source_readmes:
        source = path.read_text()
        if LOCAL_README_DUPLICATED_INVENTORY.search(source):
            failures.append(
                f"source README duplicates a file/API inventory: {relative(path)}"
            )
        if AMBIGUOUS_FLOATING_TIME_COORDINATE.search(source):
            failures.append(
                f"source README documents a floating time without units: "
                f"{relative(path)}"
            )
        if STALE_FAMILY_TAXONOMY.search(source):
            failures.append(
                f"source README documents the pre-family taxonomy: "
                f"{relative(path)}"
            )
    responsibility_paths = responsibility_inventory()
    generated_paths = {
        path for path in responsibility_paths
        if is_generated_implementation(path)
    }
    handwritten_paths = [
        path for path in responsibility_paths if path not in generated_paths
    ]
    manifest_count = sum(
        path.suffix == ".py" and "manifest" in path.stem
        for path in handwritten_paths
    )
    format_preamble_count = sum(
        responsibility_format(path, path.read_text(errors="ignore"))[1] > 0
        for path in handwritten_paths
    )
    for path in handwritten_paths:
        failure = validate_top_level_responsibility(
            path,
            path.read_text(errors="ignore"),
        )
        if failure is not None:
            failures.append(f"{failure}: {relative(path)}")

    negative_fixture = validate_top_level_responsibility(
        Path("missing_summary_fixture.cpp"),
        '#include "missing_summary_fixture.hpp"\n',
    )
    if negative_fixture is None:
        failures.append(
            "responsibility-summary negative fixture was not rejected"
        )
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix not in {".cuh", ".hpp"}:
            continue
        if AMBIGUOUS_FLOATING_TIME_COORDINATE.search(path.read_text()):
            failures.append(
                f"floating time coordinate lacks an explicit unit: "
                f"{relative(path)}"
            )
    for ambiguous_name in ("maturity", "start", "end"):
        if not AMBIGUOUS_FLOATING_TIME_COORDINATE.search(
            f"__device__ float f(float {ambiguous_name});"
        ):
            failures.append(
                "floating-time-coordinate negative fixture was missed for "
                f"{ambiguous_name}"
            )
    product_names = {
        path.name for path in (ROOT / "src" / "product").iterdir()
        if path.is_dir()
    }
    model_files = sorted(
        path for path in MODEL_ROOT.rglob("*")
        if path.is_file() and path.suffix in CPP_SUFFIXES
    )

    product_units = 0
    infrastructure_units = 0
    product_pairs: dict[tuple[str, ...], set[str]] = {}
    for path in model_files:
        parts = path.relative_to(MODEL_ROOT).parts
        boundary = model_product_boundary_index(path)
        under_product = (
            len(parts) > boundary and parts[boundary] == "product"
        )
        is_product_unit = path.stem in product_names
        failures.extend(validate_model_path(path, under_product))
        if is_product_unit:
            product_units += 1
            if not under_product:
                failures.append(
                    f"model-product unit outside product/: {relative(path)}"
                )
            product_pairs.setdefault(parts[:-1] + (path.stem,), set()).add(
                path.suffix
            )
        elif under_product:
            failures.append(
                f"non-product implementation inside product/: {relative(path)}"
            )
        else:
            infrastructure_units += 1
            source = path.read_text()
            first_line = next(
                (line.strip() for line in source.splitlines() if line.strip()),
                "",
            )
            if not first_line.startswith("// "):
                failures.append(
                    f"model infrastructure lacks a top-level summary: "
                    f"{relative(path)}"
                )
            else:
                summary = first_line[3:].strip()
                summary_words = set(re.findall(r"[a-z]+", summary.lower()))
                if not 20 <= len(summary) <= 180 or not summary.endswith("."):
                    failures.append(
                        f"model-infrastructure summary must be a short sentence: "
                        f"{relative(path)}"
                    )
                if not summary_words.intersection(SUMMARY_ROLE_TERMS):
                    failures.append(
                        f"model-infrastructure summary does not state a role: "
                        f"{relative(path)}"
                    )
                if path.name.endswith("_impl.cuh") and not (
                    summary_words.intersection(IMPLEMENTATION_HEADER_TERMS)
                ):
                    failures.append(
                        f"included implementation summary does not say what it "
                        f"defines: {relative(path)}"
                    )
                public_peer = path.with_name(f"{path.stem}_impl.cuh")
                if path.suffix == ".cuh" and public_peer.is_file() and not (
                    summary_words.intersection(PUBLIC_HEADER_TERMS)
                ):
                    failures.append(
                        f"public model header summary does not identify its "
                        f"contract role: {relative(path)}"
                    )
            if path.name not in CANONICAL_INFRASTRUCTURE_FILENAMES:
                failures.append(
                    f"unreviewed model-infrastructure filename: {relative(path)}"
                )
            if path.name in AMBIGUOUS_MODEL_FILENAMES:
                failures.append(
                    f"method-specific model helper has an ambiguous name: "
                    f"{relative(path)}"
                )

    for key, suffixes in sorted(product_pairs.items()):
        if suffixes != {".cu", ".cuh"}:
            failures.append(
                f"model-product binding lacks a .cuh/.cu pair: "
                f"{MODEL_ROOT.joinpath(*key).relative_to(ROOT).as_posix()}"
            )

    template_paths = {
        path.relative_to(TEMPLATE_ROOT).as_posix()
        for path in TEMPLATE_ROOT.rglob("*.tpl")
    }
    failures.extend(
        f"missing named codegen template: {path}"
        for path in sorted(REQUIRED_TEMPLATE_PATHS - template_paths)
    )
    failures.extend(
        f"ambiguous flat codegen template: {path}"
        for path in sorted(template_paths)
        if "/" not in path
    )

    generator_source = (
        ROOT / "tools" / "codegen" / "pricing_bindings" / "generate.py"
    ).read_text()
    if re.search(r"return\s+f?[\"']{3}// Generated", generator_source):
        failures.append(
            "generate.py contains a complete generated C++ artifact inline; "
            "move it under templates/"
        )

    product_pattern = "|".join(
        re.escape(name) for name in sorted(product_names, key=len, reverse=True)
    )
    old_equity_reference = re.compile(
        rf"model/equity/(?:markovian|rough)/[^/\s\"]+/"
        rf"(?:{product_pattern})\.(?:cu|cuh)"
    )
    old_fixed_income_reference = re.compile(
        rf"model/fixed_income/[^/\s\"]+/(?:"
        rf"(?:nelson_siegel|svensson)/)?(?:{product_pattern})\.(?:cu|cuh)"
    )
    scan_roots = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        ROOT / "catalog",
        ROOT / "validation",
        ROOT / "docs",
        ROOT / "cmake",
    )
    scan_files = [ROOT / "CMakeLists.txt", ROOT / "README.md"]
    for scan_root in scan_roots:
        scan_files.extend(
            path for path in scan_root.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
            and relative(path) != "docs/audit/closed.md"
        )
    for path in scan_files:
        source = path.read_text(errors="ignore")
        if old_equity_reference.search(source):
            failures.append(f"stale equity product path: {relative(path)}")
        if old_fixed_income_reference.search(source):
            failures.append(f"stale fixed-income product path: {relative(path)}")

    if failures:
        print("\n".join(sorted(set(failures))))
        return 1
    print(
        f"{product_units} model-product files isolated under product/; "
        f"{infrastructure_units} infrastructure files summarized; "
        f"{len(template_paths)} named codegen templates; "
        f"{len(generated_paths)} generated and "
        f"{len(handwritten_paths)} handwritten files classified "
        f"({manifest_count} manifests, {format_preamble_count} format preambles)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
