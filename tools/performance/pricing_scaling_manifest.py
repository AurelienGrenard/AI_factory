"""Derive representative scaling workloads from the canonical pricing capabilities.

Only product selection and workload sizes belong here; model identities, paths,
engines, seeds and launcher signatures remain owned by the codegen manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from string import Template

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/codegen/pricing_bindings"))
from capability_manifest import (  # noqa: E402
    MODEL_SPECS, MODEL_BY_NAME, PRODUCT_BINDING_SPECS, DATASET_SPECS,
    CURVE_BY_NAME, resolve_rng_domain,
)

TEMPLATES = ROOT / "tests/performance/pricing_scaling"
PATH_COUNTS = (65536, 262144, 1048576)
PRICE_COUNTS = (100, 1000, 10000)


def public_launcher(header_path: str) -> tuple[str, list[str]]:
    header = (ROOT / header_path).read_text()
    declaration = re.search(r"(?:void|longstaff_schwartz::LaunchResult)\s+(launch_\w+_cuda)\s*\((.*?)\);", header, re.S)
    if declaration is None:
        raise ValueError(f"No public declaration: {header_path}")
    launcher, parameters = declaration.groups()
    names = [re.search(r"(\w+)\s*$", p.strip()).group(1) for p in parameters.split(",")]
    return launcher, names


def workloads(include_lsm: bool = False) -> list[dict]:
    selected = []
    for binding in PRODUCT_BINDING_SPECS:
        family = None
        variant = None
        if include_lsm and "lsm" in binding.engine:
            family = "lsm"
            variant = ("american_puts" if binding.asset_class == "equity"
                       else "bermudan_payer_swaptions")
        elif binding.asset_class == "equity":
            if binding.product == "european_option":
                family = ("closed_form" if binding.engine == "equity_closed_form"
                          else "mc_terminal")
                variant = "european_calls"
            elif binding.product == "down_and_out_option":
                family, variant = "mc_barrier", "down_and_out_puts"
        elif binding.product == "rate_option":
            family, variant = "closed_form", "caplets"
        if family is None:
            continue
        recipes = [d for d in DATASET_SPECS if d.dataset_kind == "prices"
                   and d.construction == "aligned"
                   and d.model == binding.model and d.curve == binding.curve
                   and d.product == binding.product and d.variant == variant]
        if len(recipes) != 1 and not (family == "lsm" and not recipes):
            raise ValueError(f"Expected one recipe: {binding}, got {len(recipes)}")
        recipe = recipes[0] if recipes else None
        model = MODEL_BY_NAME[binding.model]
        product_datasets = [d for d in DATASET_SPECS
                            if d.dataset_kind == "product_parameters"
                            and d.product == binding.product]
        if len(product_datasets) != 1:
            raise ValueError(f"Ambiguous product dataset: {binding.product}")
        key = "__".join(filter(None, (binding.model, binding.curve, family)))
        selected.append({
            "id": key, "family": family, "model": binding.model,
            "curve": binding.curve, "product": binding.product,
            "engine": binding.engine,
            "side": ("payer" if variant == "bermudan_payer_swaptions" else
                     "put" if family in ("mc_barrier", "lsm") else "call"),
            "model_prefix": model.source_prefix,
            "model_dataset": f"datasets/{model.source_prefix}/parameters/{model.parameter_dataset_id}.json",
            "product_dataset": product_datasets[0].dataset_path,
            "generator": recipe.generator_path if recipe else None, "header": binding.paths[0],
            "scope": "catalogue_recipe" if recipe else "binding_only_no_price_recipe",
            "target": "ai_factory_" + binding.asset_class + "_" + "_".join(
                filter(None, (binding.model, binding.curve, binding.product))),
            "seed": (0 if family == "closed_form" else
                     resolve_rng_domain(recipe).seed("dynamics") if recipe else
                     int.from_bytes(hashlib.sha256(("scaling-only/" + key).encode()).digest()[:7], "big")),
            "seed_source": "canonical_recipe" if recipe else "scaling_only_sha256_case_id_56bit",
            "time_kind": (None if family == "closed_form" else "fixed"
                          if {"dt", "target_dt"}.intersection(public_launcher(binding.paths[0])[1])
                          else "exact"),
            "factor_count": 7 if binding.engine == "equity_n_factor" else None,
            "mathdx": binding.engine == "equity_volterra_fft",
        })
    return sorted(selected, key=lambda item: item["id"])


def coverage(cases: list[dict]) -> list[dict]:
    return [{"model": m.name, "source_prefix": m.source_prefix,
             "configurations": {
                 family: [w["id"] for w in cases if w["model"] == m.name and w["family"] == family]
                 or "not_applicable_no_published_binding"
                 for family in ("mc_terminal", "mc_barrier", "closed_form", "lsm")},
             "lsm": "available_separate_campaign" if any(
                 b.model == m.name and "lsm" in b.engine for b in PRODUCT_BINDING_SPECS
             ) else "not_applicable"}
            for m in MODEL_SPECS]


def render_case(case: dict) -> str:
    lsm = case["family"] == "lsm"
    launcher, names = public_launcher(case["header"])
    expressions = {
        "device_models": "device_models.template as<Model>()", "model_count": "models.size()",
        "host_products": "products.data()", "device_products": "device_products.template as<Product>()",
        "product_count": "products.size()", "construction": "PriceConstruction::Aligned",
        "result_count": "models.size()", "result_offset": "offset", "launch_result_count": "count",
        "monte_carlo_paths_per_price": "job.paths", "dt": "kDt",
        "simulation_steps_per_day": "kStepsPerDay", "day_fraction": "kDayFraction",
        "threads_per_block": "job.threads", "block_count": "blocks",
        "base_seed": "seed", "device_prices": "device_prices.template as<float>()",
        "device_standard_errors": "device_errors.template as<float>()",
        "device_prepared_dynamics": "device_prepared.template as<Prepared>()",
        "prepared_dynamics_count": "prepared.size()",
        "result_index": "offset", "target_dt": "kDt",
        "step_count": "static_cast<std::size_t>(products[offset].maturity_days) * kStepsPerDay",
        "path_chunk_size": "job.path_chunk", "device_workspace": "workspace.template as<void>()",
        "workspace_bytes": "workspace_bytes",
        "device_curves": "device_curves.template as<Curve>()", "curve_count": "curves.size()",
        "time_day_fraction": "kDayFraction", "blocks_per_price": "job.blocks_per_price",
    }
    if lsm:
        # Published LSM APIs operate on a complete aligned slice. Translate the
        # global row offset exactly once for both pointers and the row key.
        for name in ("device_models", "host_products", "device_products",
                     "device_curves", "device_prices", "device_standard_errors"):
            expressions[name] += " + offset"
        for name in ("model_count", "product_count", "curve_count", "result_count"):
            expressions[name] = "count"
        expressions["base_seed"] = "seed + offset"
    unknown = set(names) - expressions.keys()
    if unknown:
        raise ValueError(f"Unsupported public arguments {unknown}: {case['id']}")
    model_namespace = ("model::" + MODEL_BY_NAME[case["model"]].asset_class + "::" + case["model"])
    launch_namespace = "model_binding" + ("::" + case["curve"] if case["curve"] else "")
    product_type, loader = {
        "european_option": ("EuropeanOptionParameters", "load_european_options"),
        "down_and_out_option": ("DownAndOutOptionParameters", "load_down_and_out_options"),
        "rate_option": ("RateOptionParameters", "load_rate_options"),
        "american_option": ("AmericanOptionParameters", "load_american_options"),
        "bermudan_swaption": ("BermudanSwaptionParameters", "load_bermudan_swaptions"),
    }[case["product"]]
    auxiliary_kind = ("n_factor" if case["factor_count"] else "volterra_fft" if case["mathdx"]
                      else "curve" if case["curve"] else "none")
    extra_include = ""
    if case["factor_count"]:
        extra_include = f'#include "{case["model_prefix"]}/markovian_n_factor_preparation.hpp"'
    if case["curve"]:
        extra_include = f'#include "curve/{case["curve"]}/dataset.hpp"'
    values = {
        "HEADER": case["header"].removeprefix("src/"), "MODEL_PREFIX": case["model_prefix"],
        "PRODUCT": case["product"], "PRODUCT_TYPE": product_type,
        "MODEL_NAMESPACE": model_namespace, "EXTRA_INCLUDE": extra_include,
        "MODEL_DATASET": case["model_dataset"], "PRODUCT_DATASET": case["product_dataset"],
        "PRODUCT_LOADER": loader, "SEED": str(case["seed"]), "CASE_ID": case["id"],
        "CLOSED_FORM": str(case["family"] == "closed_form").lower(),
        "VOLTERRA": str(case["mathdx"]).lower(),
        "LAUNCHER": f"{launch_namespace}::{launcher}",
        "TEMPLATE_ARGS": ("SwaptionSide::" if case["product"] == "bermudan_swaption"
                          else "OptionSide::") + case["side"] + (", 7U" if case["factor_count"] else ""),
        "RETURN": "return " if lsm else "",
        "TIME_STEP_DESCRIPTION": "1 / 504" if {"dt", "target_dt"}.intersection(names) else "",
        "LAUNCH_ARGS": ",\n                    ".join(expressions[n] for n in names),
        "CURVE": case["curve"] or "", "CURVE_TYPE": (
            CURVE_BY_NAME[case["curve"]].cpp_type + "Parameters" if case["curve"] else ""),
        "CURVE_DATASET": (f'datasets/curve/{case["curve"]}/{case["curve"]}_01.json'
                          if case["curve"] else ""),
    }
    auxiliary = ""
    if auxiliary_kind != "none":
        auxiliary = Template((TEMPLATES / f"{auxiliary_kind}_inputs.cuh.tpl").read_text()).substitute(values)
    values["AUXILIARY_INPUTS"] = auxiliary
    return Template((TEMPLATES / "benchmark.cu.tpl").read_text()).substitute(values)


def generate(output: Path, mathdx: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = workloads(include_lsm=True)
    active = [c for c in cases if mathdx or not c["mathdx"]]
    cmake = ["# Generated scaling probes, derived from the canonical capability manifest."]
    for case in active:
        source = output / (case["id"] + ".cu")
        rendered = render_case(case)
        if not source.exists() or source.read_text() != rendered:
            source.write_text(rendered)
        target = "ai_factory_pricing_scaling_" + case["id"]
        dependencies = [case["target"],
                        f'ai_factory_{MODEL_BY_NAME[case["model"]].asset_class}_{case["model"]}_dataset',
                        f'ai_factory_product_{case["product"]}_dataset',
                        "ai_factory_runtime", "ai_factory_price_dataset"]
        if case["curve"]:
            dependencies.append(f'ai_factory_curve_{case["curve"]}_dataset')
        if case["mathdx"]:
            dependencies.append("ai_factory_cufftdx")
        cmake += [f'add_executable({target} EXCLUDE_FROM_ALL "{source}")',
                  f'ai_factory_configure_cuda_library({target})',
                  f'target_link_libraries({target} PRIVATE {" ".join(dependencies)})',
                  f'add_dependencies(pricing_scaling_benchmarks {target})']
    (output / "targets.cmake").write_text("\n".join(cmake) + "\n")
    manifest = {"schema": "ai_factory_pricing_scaling_v2", "cases": cases,
                "coverage": coverage(cases), "path_counts": PATH_COUNTS,
                "price_counts": PRICE_COUNTS, "production_price_target": 1000000,
                "exclusions": ["samples", "independent_validation"],
                "execution_order": ["mc_terminal", "mc_barrier", "closed_form", "lsm"],
                "geometry_selection": "per model, engine, price count and path count; never one universal configuration",
                "not_built": [c["id"] for c in cases if c not in active]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mathdx", action="store_true")
    arguments = parser.parse_args()
    manifest = generate(arguments.output.resolve(), arguments.mathdx)
    print(f"Scaling: {len(manifest['cases'])} cases, {len(manifest['not_built'])} unavailable")
