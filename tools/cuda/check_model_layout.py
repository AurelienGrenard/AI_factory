#!/usr/bin/env python3
"""Enforce model/product ownership, file summaries and codegen discoverability."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "model"
TEMPLATE_ROOT = ROOT / "tools" / "codegen" / "pricing_bindings" / "templates"
CPP_SUFFIXES = {".cu", ".cuh", ".cpp", ".hpp"}
TEXT_SUFFIXES = CPP_SUFFIXES | {".cmake", ".md", ".py", ".txt"}
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
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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
        f"{len(template_paths)} named codegen templates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
