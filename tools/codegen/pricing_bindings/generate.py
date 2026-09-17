#!/usr/bin/env python3
"""Generate thin model-product pricing bindings into an output folder."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
from string import Template

from manifest import (
    AMERICAN_RECIPE_SPECS,
    BINDINGS,
    BLACK_SCHOLES_CLOSED_FORM_PRODUCTS,
    MARKOVIAN_MODELS,
    MODEL_RECIPE_SPECS,
    PRICE_VARIANTS,
    ROUGH_PRODUCT_BINDINGS,
    ROUGH_MODELS,
    ROUGH_N_FACTOR_MODELS,
    ROUGH_VOLTERRA_MODELS,
    Binding,
    RoughProductBinding,
)
from capability_manifest import (
    AVAILABLE_DATASET_SPECS,
    CAPABILITY_EXCEPTIONS,
    CURVE_BY_NAME,
    CURVE_SPECS,
    DEFERRED_DATASET_SPECS,
    DECLARED_PRODUCT_BINDING_PATHS,
    ENGINE_SPECS,
    EQUITY_EARLY_EXERCISE_UNITS,
    EQUITY_MATHDX_SAMPLE_UNITS,
    EQUITY_SAMPLE_UNITS,
    FIXED_INCOME_CAPABILITIES,
    FIXED_INCOME_UNITS,
    FIXED_INCOME_VARIANTS,
    GENERATED_PRODUCT_BINDING_SPECS,
    GENERATED_PRODUCT_BINDING_PATHS,
    MODEL_BY_NAME,
    MODEL_SPECS,
    PRODUCT_SPECS,
    PRODUCT_BINDING_SPECS,
    PRICE_DELTA_BINDING_SPECS,
    PRICE_GRADIENT_BINDING_SPECS,
    GENERATED_PRICE_GRADIENT_BINDING_PATHS,
    PRICE_GRADIENT_DATASET_SPECS,
    PRICE_GRADIENT_SOURCE_BY_GENERATOR,
    GENERATED_PRICE_DELTA_BINDING_PATHS,
    GENERATED_CLOSED_FORM_POLICY_PATHS,
    PriceDeltaBindingSpec,
    PRICE_DELTA_DATASET_SPECS,
    PRICE_DELTA_SOURCE_BY_GENERATOR,
    SCHEMA_VERSION,
    pricing_launch_family,
    resolve_rng_domain,
)
from sample_manifest import SAMPLE_MODELS, SAMPLE_MODEL_BY_NAME, SampleModelSpec
from price_gradients.render import render_bindings as render_price_gradient_bindings
from price_gradients.render import render_recipes as render_price_gradient_recipes


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


def _write_generated(path: Path, contents: str) -> None:
    """Keep unchanged outputs' timestamps so regeneration does not rebuild CUDA."""
    if not path.exists() or path.read_text() != contents:
        path.write_text(contents)


def _sample_namespace(model: SampleModelSpec) -> str:
    return f"model::{model.asset_class}::{model.name}"


def _sample_observation(model: SampleModelSpec, dynamics: str) -> str:
    parts = model.observation.split(":")
    if parts[0] == "spot":
        return f"sample::SpotSampleObservation<{dynamics}>"
    if parts[0] == "state":
        return f"sample::StateSampleObservation<{dynamics}>"
    if parts[0] == "spot_state":
        return (
            f"sample::SpotAndStateSampleObservation<{dynamics}, "
            f"&State:: {parts[1]}>"
        ).replace(":: ", "::")
    if parts[0] == "two_state":
        return (
            f"sample::TwoStateMemberSampleObservation<{dynamics}, "
            f"&State::{parts[1]}, &State::{parts[2]}>"
        )
    raise ValueError(f"Unsupported sample observation: {model.observation}")


def _sample_output_declarations(model: SampleModelSpec) -> str:
    return "".join(
        f",\n    float* device_{_sample_output_pointer_name(name)}"
        for name in model.outputs
    )


def _sample_output_values(model: SampleModelSpec) -> str:
    return ", ".join(
        f"device_{_sample_output_pointer_name(name)}"
        for name in model.outputs
    )


def _sample_output_pointer_name(name: str) -> str:
    return {
        "spot": "spots",
        "state": "states",
        "state_x": "states_x",
        "state_y": "states_y",
        "variance": "variances",
        "reciprocal_variance": "reciprocal_variances",
        "volatility": "volatilities",
        "alpha": "alphas",
    }.get(name, name + "s")


def _render_dollar_template(relative_path: str, values: dict[str, str]) -> str:
    """Render a named template without making C++ braces special."""
    template = Template((TEMPLATE_DIR / relative_path).read_text())
    return template.substitute(values)


def _render_cpp_fragment_template(
    relative_path: str,
    values: dict[str, str],
) -> str:
    """Render a documented C++ fragment without copying its file summary."""
    rendered = _render_dollar_template(relative_path, values)
    summary, separator, fragment = rendered.partition("\n")
    if not separator or not summary.startswith("// "):
        raise ValueError(f"C++ fragment template lacks a summary: {relative_path}")
    return fragment


def _sample_binding_template_values(model: SampleModelSpec) -> dict[str, str]:
    declarations = _sample_output_declarations(model)
    values = {
        "display": model.display,
        "model_name": model.name,
        "namespace": _sample_namespace(model),
        "source_folder": model.source_folder,
        "output_declarations": declarations,
        "output_declarations_inline": declarations.lstrip(),
        "output_values": _sample_output_values(model),
        "output_types": ", float*" * len(model.outputs),
        "schedule_prefix": (
            "ExactTransition" if model.time_kind == "exact" else "FixedStep"
        ),
        "dynamics_header": (
            f"model/{model.source_folder}/dynamics.cuh"
            if model.extra_dynamics_include
            else f"model/{model.source_folder}/dynamics_impl.cuh"
        ),
        "extra_include": model.extra_dynamics_include,
        "observation": _sample_observation(model, "DynamicsPolicy"),
        "kernel": model.kernel or "",
        "kernel_header": (
            _volterra_kernel_header(model.kernel) if model.kernel else ""
        ),
    }
    return values


def _render_sample_binding(model: SampleModelSpec, suffix: str) -> str:
    family = {
        "markovian": "markovian",
        "volterra": "rough/volterra_fft",
        "n_factor": "rough/markovian_n_factor",
    }[model.backend]
    return _render_dollar_template(
        f"sampling/{family}/model_binding.{suffix}.tpl",
        _sample_binding_template_values(model),
    )



def _volterra_kernel_header(kernel: str) -> str:
    return {
        "volterra::FractionalHybridKernelPolicy":
            "common/volterra/fractional_hybrid_kernel.cuh",
        "volterra::LogModulatedHybridKernelPolicy":
            "common/volterra/log_modulated_hybrid_kernel.cuh",
        "volterra::FractionalResolventHybridKernelPolicy":
            "common/volterra/fractional_resolvent_hybrid_kernel.cuh",
    }[kernel]




def _cpp_float(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def _sample_bounds_json(model: SampleModelSpec) -> str:
    entries = ",\n                ".join(
        '{"' + name + '", {' + _cpp_float(minimum) + ', '
        + _cpp_float(maximum) + '}}'
        for name, minimum, maximum in model.uniforms
    )
    return "{\n                " + entries + "\n            }"


def _sample_parameter_factory(model: SampleModelSpec) -> str:
    uniforms = "\n        ".join(
        f"const float {name} = uniform({{{_cpp_float(minimum)}, "
        f"{_cpp_float(maximum)}}}, uniforms);"
        for name, minimum, maximum in model.uniforms
    )
    derived = f"\n        {model.derived}" if model.derived else ""
    return _render_cpp_fragment_template(
        "sampling/catalog/model_parameter_factory.cuh.tpl",
        {
            "uniforms": uniforms,
            "derived": derived,
            "acceptance": model.acceptance,
            "constructor": model.constructor,
        },
    ).rstrip()


def _sample_parameter_json(model: SampleModelSpec) -> str:
    entries = ",\n        ".join(
        f'{{"{name}", parameters.{accessor}}}'
        for name, accessor in model.parameters
    )
    return _render_cpp_fragment_template(
        "sampling/catalog/model_parameter_json.cuh.tpl",
        {"entries": entries},
    ).rstrip()


def _sample_launch_lambda(model: SampleModelSpec) -> str:
    pointers = ",\n            ".join(
        f"outputs[{index}]" for index in range(len(model.outputs))
    )
    launcher = f"model_binding::launch_{model.name}_random_terminal_samples_cuda"
    if model.backend == "volterra":
        geometry = "block_count,"
    else:
        geometry = "threads_per_block, block_count,"
    template = "<factor_count>" if model.backend == "n_factor" else ""
    return _render_cpp_fragment_template(
        "sampling/catalog/sample_launch_lambda.cuh.tpl",
        {
            "output_count": f"{len(model.outputs)}U",
            "display": model.display,
            "launcher": launcher,
            "template_arguments": template,
            "geometry": geometry,
            "thread_argument": (
                "/* cuFFTDx fixes the block dimensions */"
                if model.backend == "volterra" else "threads_per_block"
            ),
            "output_pointers": pointers,
        },
    ).rstrip()




def _render_sample_generation_header(model: SampleModelSpec) -> str:
    include_numerics = ""
    generate_call = "generate_model_sample_dataset<ModelParameters>"
    prepare_argument = ""
    preparation_metadata = ""
    if model.backend == "n_factor":
        include_numerics = (
            f'#include "model/{model.source_folder}/markovian_n_factor_preparation.hpp"\n'
        )
        type_arguments = (
            "ModelParameters, model_binding::PreparedDynamics<factor_count>"
        )
        generate_call = (
            f"generate_prepared_model_sample_dataset<{type_arguments}>"
        )
        prepare_argument = "\n        " + _render_cpp_fragment_template(
            "sampling/catalog/preparation/markovian_n_factor_lambda.cuh.tpl",
            {},
        ).rstrip().replace("\n", "\n        ")
    if model.name == "quadratic_rough_heston":
        prepare_argument = "\n        " + _render_cpp_fragment_template(
            "sampling/catalog/preparation/quadratic_rough_heston_lambda.cuh.tpl",
            {},
        ).rstrip().replace("\n", "\n        ")
        preparation_metadata = _render_cpp_fragment_template(
            "sampling/catalog/preparation/quadratic_rough_heston_metadata.cuh.tpl",
            {},
        ).rstrip().replace("\n", "\n            ")
    numerical = {
        ("markovian", "exact"): "exact finite-horizon transition",
        ("markovian", "fixed"): "fixed-step transition at dt=1/504",
        ("volterra", "fixed"): "block-cooperative hybrid FFT at dt=1/504",
        ("n_factor", "fixed"): "seven-factor Markovian lift at dt=1/504",
    }[(model.backend, model.time_kind)]
    grid = (
        '{{"transition", "exact"}, {"delta_t", "maturity_days / 252"}, '
        '{"artificial_substeps", false}}'
        if model.time_kind == "exact" else
        '{{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, '
        '{"simulation_steps_per_day", 2}}'
    )
    descriptions = dict(model.observable_descriptions)
    output_metadata = ",\n            ".join(
        f'{{"{name}", {{{{"description", {json.dumps(descriptions.get(name, f"Terminal {name}."))}}}, '
        f'{{"layout", "sample-major"}}}}}}' for name in model.outputs
    )
    return _render_dollar_template(
        "sampling/catalog/recipe_support.cuh.tpl",
        {
            "display": model.display,
            "source_folder": model.source_folder,
            "include_numerics": include_numerics,
            "model_name": model.name,
            "namespace": _sample_namespace(model),
            "parameter_factory": _sample_parameter_factory(model),
            "parameter_json": _sample_parameter_json(model),
            "asset_class": model.asset_class,
            "numerical": numerical,
            "sample_bounds": _sample_bounds_json(model),
            "proposal_draw_order": ", ".join(json.dumps(name) for name, _, _ in model.uniforms),
            "escaped_acceptance": model.acceptance.replace('"', '\\"'),
            "derived_parameter_metadata": (
                ',\n            {"derived_parameters", {'
                + ", ".join(
                    f'{{{json.dumps(name)}, {json.dumps(law)}}}'
                    for name, law in model.derived_parameter_laws
                ) + "}}"
                if model.derived_parameter_laws else ""
            ),
            "output_metadata": output_metadata,
            "grid": grid,
            "generate_call": generate_call,
            "backend_samples": f"{model.backend}_samples",
            "native_block_declaration": (
                f"    const auto native_block = model_binding::{model.name}_sample_block_dimensions(value.maximum_maturity_days);"
                if model.backend == "volterra" else ""
            ),
            "native_block_argument": ", native_block" if model.backend == "volterra" else "",
            "output_names": ", ".join(
                f'"{name}"' for name in model.outputs
            ),
            "prepare_argument": prepare_argument,
            "preparation_metadata": preparation_metadata,
            "launch_lambda": _sample_launch_lambda(model),
        },
    )


def _render_sample_recipe_source(
    model: SampleModelSpec,
    recipe_index: int,
) -> str:
    if recipe_index == 1:
        parameter_count, paths_per_parameter = "12'000U", "250U"
        sample_kind = "conditional"
    else:
        parameter_count, paths_per_parameter = "3'000'000U", "1U"
        sample_kind = "unconditional"
    recipe_path = (
        f"catalog/model/{model.source_folder}/samples/"
        f"samples_{recipe_index:02d}/generator.cpp"
    )
    domain = resolve_rng_domain(recipe_path)
    return _render_dollar_template(
        "sampling/catalog/generator.cpp.tpl",
        {
            "display": model.display,
            "sample_kind": sample_kind,
            "source_folder": model.source_folder,
            "model_name": model.name,
            "database_id": f"samples_{recipe_index:02d}",
            "parameter_count": parameter_count,
            "paths_per_parameter": paths_per_parameter,
            "parameter_seed": str(domain.seed("parameters")),
            "schedule_seed": str(domain.seed("schedule")),
            "dynamics_seed": str(domain.seed("dynamics")),
        },
    )


def _sample_recipe_metadata(
    model: SampleModelSpec,
    recipe_index: int,
    dataset,
) -> dict:
    parameter_count, paths_per_parameter = (
        (12_000, 250) if recipe_index == 1 else (3_000_000, 1)
    )
    domain = resolve_rng_domain(dataset)
    descriptions = dict(model.observable_descriptions)
    return {
        "schema_version": 1,
        "kind": "samples",
        "dataset_id": dataset.dataset_id,
        "generator": "generator.cpp",
        "model": model.name,
        "output": {
            "path": dataset.dataset_path,
            "format": "json",
        },
        "generation_output": dataset.generation_yaml_path,
        "shape": {
            "parameter_count": parameter_count,
            "paths_per_parameter": paths_per_parameter,
            "row_count": parameter_count * paths_per_parameter,
            "row_order": "parameter-major, then path-major",
        },
        "parameter_sampling": {
            "regime": "plausible core only",
            "distribution": (
                "independent Philox uniform proposals; accepted rows retain "
                "proposal order"
            ),
            "proposal_draw_order": [name for name, _, _ in model.uniforms],
            "latent_uniform_bounds": {
                name: [minimum, maximum]
                for name, minimum, maximum in model.uniforms
            },
            "acceptance": model.acceptance,
            "derived_parameters": dict(model.derived_parameter_laws),
        },
        "maturity_sampling": {
            "distribution": "discrete uniform without modulo bias",
            "support": "integer business days",
            "minimum_days": 63,
            "maximum_days": 504,
            "year_fraction": "maturity_days / 252",
        },
        "seeds": {
            name: domain.seed(name) for name in domain.streams
        },
        "numerical_method": {
            "engine": dataset.engine,
            "profile": dataset.numerical_profile,
            "description": model.pricing_numerical_method,
        },
        "outputs": {
            name: {
                "description": descriptions.get(name, f"Terminal {name}."),
                "layout": "sample-major",
            }
            for name in model.outputs
        },
        "time_grid": (
            {
                "transition": "exact",
                "delta_t": "maturity_days / 252",
                "artificial_substeps": False,
            }
            if model.time_kind == "exact"
            else {
                "transition": "fixed-step",
                "delta_t": "1 / 504",
                "simulation_steps_per_day": 2,
            }
        ),
    }


def generate_samples(output_root: Path) -> list[Path]:
    generated: list[Path] = []
    for model in SAMPLE_MODELS:
        source_directory = output_root / "src" / "model" / model.source_folder
        source_directory.mkdir(parents=True, exist_ok=True)
        header = source_directory / "sample.cuh"
        source = source_directory / "sample.cu"
        _write_generated(header, _render_sample_binding(model, "cuh"))
        _write_generated(source, _render_sample_binding(model, "cu"))
        generated.extend((header, source))

        helper = (
            output_root / "tools" / "sampling" / "generated"
            / f"{model.name}_sample_generation.cuh"
        )
        helper.parent.mkdir(parents=True, exist_ok=True)
        _write_generated(helper, _render_sample_generation_header(model))
        generated.append(helper)
        for recipe_index in (1, 2):
            recipe = (
                output_root / "catalog" / "model" / model.source_folder
                / "samples" / f"samples_{recipe_index:02d}"
                / "generator.cpp"
            )
            recipe.parent.mkdir(parents=True, exist_ok=True)
            _write_generated(recipe, _render_sample_recipe_source(model, recipe_index))
            generated.append(recipe)
            dataset = next(
                item for item in AVAILABLE_DATASET_SPECS
                if item.dataset_kind == "samples"
                and item.model == model.name
                and item.dataset_id == f"samples_{recipe_index:02d}"
            )
            metadata_path = recipe.with_name("recipe.yaml")
            _write_generated(
                metadata_path,
                json.dumps(
                    _sample_recipe_metadata(model, recipe_index, dataset),
                    indent=2,
                ) + "\n",
            )
            generated.append(metadata_path)
    return generated


def time_fields(binding: Binding) -> dict[str, str]:
    if binding.time_kind == "exact":
        return {
            "time_parameter_declarations": "    float day_fraction,\n",
            "time_configuration": (
                "    const simulation::ExactTransitionTimeConfiguration "
                "time_configuration{\n"
                "        day_fraction,\n"
                "    };\n"
            ),
            "instantiation_time_signature": (
                "    float, unsigned int, std::size_t, std::uint64_t,"
            ),
        }
    if binding.time_kind == "fixed":
        return {
            "time_parameter_declarations": (
                "    float dt,\n"
                "    std::uint32_t simulation_steps_per_day,\n"
            ),
            "time_configuration": (
                "    const simulation::FixedStepTimeConfiguration "
                "time_configuration{\n"
                "        dt,\n"
                "        simulation_steps_per_day,\n"
                "    };\n"
            ),
            "instantiation_time_signature": (
                "    float, std::uint32_t, unsigned int, std::size_t, "
                "std::uint64_t,"
            ),
        }
    raise ValueError(f"Unsupported time kind: {binding.time_kind}")


def render(template: str, binding: Binding) -> str:
    product_words = binding.product.replace("_", " ")
    schedule_expression = (
        f"{binding.schedule}<{binding.model}::DynamicsPolicy, 2U>"
        if binding.schedule.endswith("CalendarSchedule")
        else f"{binding.schedule}<{binding.model}::DynamicsPolicy>"
    )
    template_declaration = "template<OptionSide Side>\n" if binding.sided else ""
    pricing_policy_declaration = (
        "template<OptionSide Side>\nusing PricingPolicy =\n"
        f"    product::{binding.pricing_policy}<Schedule, Side>;"
        if binding.sided
        else f"using PricingPolicy = product::{binding.pricing_policy}<Schedule>;"
    )
    pricing_policy_use = "PricingPolicy<Side>" if binding.sided else "PricingPolicy"
    static_pricing_policy_use = (
        "PricingPolicy<OptionSide::call>" if binding.sided else "PricingPolicy"
    )
    diagnostic_variant = "option_side_name(Side)" if binding.sided else '"default"'
    explicit_instantiations = ""
    if binding.sided:
        signature = time_fields(binding)["instantiation_time_signature"]
        explicit_instantiations = "\n".join(
            f"""template void launch_{binding.model}_{binding.product}_cuda<OptionSide::{side}>(
    const ModelParameters*, std::size_t,
    const product::{binding.product_type}Parameters*,
    const product::{binding.product_type}Parameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
{signature}
    float*, float*
);"""
            for side in ("call", "put")
        )
    values = {
        **binding.__dict__,
        **time_fields(binding),
        "product_comment": binding.product.replace("_", "-"),
        "header_product_comment": binding.product.replace(
            "_", "-"
        ).capitalize(),
        "operation_product": product_words,
        "schedule_expression": schedule_expression,
        "template_declaration": template_declaration,
        "pricing_policy_declaration": pricing_policy_declaration,
        "pricing_policy_use": pricing_policy_use,
        "static_pricing_policy_use": static_pricing_policy_use,
        "diagnostic_variant": diagnostic_variant,
        "explicit_instantiations": explicit_instantiations,
    }
    return template.format(**values)


def generate_markovian(output_root: Path) -> list[Path]:
    binding_template_dir = TEMPLATE_DIR / "pricing" / "markovian"
    header_template = (binding_template_dir / "product_binding.cuh.tpl").read_text()
    source_template = (binding_template_dir / "product_binding.cu.tpl").read_text()
    generated: list[Path] = []
    for spec in GENERATED_PRODUCT_BINDING_SPECS:
        if spec.engine != "equity_markovian":
            continue
        binding = spec.manifest_binding
        if not isinstance(binding, Binding):
            raise TypeError(f"missing markovian Binding: {spec.unit_path}")
        destination = (output_root / spec.unit_path).parent
        destination.mkdir(parents=True, exist_ok=True)
        header = destination / f"{binding.product}.cuh"
        source = destination / f"{binding.product}.cu"
        _write_generated(header, render(header_template, binding))
        _write_generated(source, render(source_template, binding))
        generated.extend((header, source))
    analytical_template_dir = (
        TEMPLATE_DIR / "pricing" / "closed_form" / "black_scholes"
    )
    for spec in GENERATED_PRODUCT_BINDING_SPECS:
        if spec.engine != "equity_closed_form":
            continue
        analytical_destination = (output_root / spec.unit_path).parent
        analytical_destination.mkdir(parents=True, exist_ok=True)
        for suffix in ("cuh", "cu"):
            destination = analytical_destination / f"{spec.product}.{suffix}"
            template = (
                analytical_template_dir / f"{spec.product}.{suffix}.tpl"
            )
            _write_generated(destination, template.read_text())
            generated.append(destination)
        if spec.product != "european_option":
            destination = analytical_destination / f"{spec.product}_impl.cuh"
            _write_generated(destination, (analytical_template_dir / f"{spec.product}_impl.cuh.tpl").read_text())
            generated.append(destination)
    return generated


def closed_form_delta_values(spec: PriceDeltaBindingSpec) -> dict[str, str]:
    """Reuse the exact price policy, including its contractual calendar guard."""
    product = next(p for p in ROUGH_PRODUCT_BINDINGS if p.product == spec.pricing.product)
    european = product.product == "european_option"
    fixed = product.product == "geometric_asian_option"
    sided = product.sided
    time_types = "float, std::uint32_t" if fixed else "float"
    host_type = f"const product::{product.product_type}Parameters*, " if fixed else ""
    signature = (
        "const ModelParameters*, const ModelParameters*, std::size_t, "
        f"{host_type}const product::{product.product_type}Parameters*, std::size_t, PriceConstruction, "
        f"std::size_t, std::size_t, std::size_t, {time_types}, unsigned int, std::size_t, "
        "::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*")
    policy = ("product::LognormalEuropeanOptionClosedFormPolicy<ModelParameters, Side>" if european
              else f"{product.product_type}ClosedFormPricingPolicy" + ("<Side>" if sided else ""))
    guard = ""
    if fixed:
        guard = "validate_geometric_asian_calendar(host_products, product_count, construction, result_count, {dt, simulation_steps_per_day});"
    return {
        "product": product.product, "product_type": product.product_type,
        "side_declaration": "template<OptionSide Side>" if sided else "",
        "host_product_declaration": f"const product::{product.product_type}Parameters* host_products," if fixed else "",
        "time_declaration": "float dt, std::uint32_t simulation_steps_per_day" if fixed else "float day_fraction",
        "time_configuration": "{dt, simulation_steps_per_day}" if fixed else "{day_fraction}",
        "calendar_guard": guard, "price_policy": policy,
        "policy_include": ('#include "model/equity/markovian/black_scholes/analytics_impl.cuh"\n'
                           '#include "product/european_option/closed_form_pricing_policy.cuh"' if european else
                           f'#include "model/equity/markovian/black_scholes/product/{product.product}_impl.cuh"'),
        "diagnostic_variant": "option_side_name(Side)" if sided else '"default"',
        "instantiations": "\n".join(
            f"template void launch_black_scholes_{product.product}_price_delta_cuda<OptionSide::{side}>(\n    {signature});"
            for side in (("call", "put") if sided else ())),
    }


def generate_price_delta_bindings(output_root: Path) -> list[Path]:
    """Compose spot-delta strategies without copying any model or payoff body."""
    generated = []
    for spec in PRICE_DELTA_BINDING_SPECS:
        if spec.pricing.engine == "equity_volterra_fft":
            binding = spec.pricing.manifest_binding
            values = rough_values(binding, spec.pricing.model,
                                  MODEL_BY_NAME[spec.pricing.model].display, "volterra")
            sample = SAMPLE_MODEL_BY_NAME[spec.pricing.model]
            values.update(kernel=sample.kernel, kernel_header=_volterra_kernel_header(sample.kernel),
                          path_strategy=("CoupledVolterraSpotPaths" if spec.path_strategy == "coupled"
                                         else "MultiplicativeVolterraSpotPath"))
            values["explicit_instantiations"] = values["explicit_instantiations"].replace(
                f"launch_{spec.pricing.model}_{binding.product}_cuda",
                f"launch_{spec.pricing.model}_{binding.product}_price_delta_cuda"
            ).replace("const ModelParameters*, std::size_t,",
                      "const ModelParameters*, const ModelParameters*, std::size_t,").replace(
                f"const product::{binding.product_type}Parameters*, std::size_t,",
                f"const product::{binding.product_type}Parameters*, const product::{binding.product_type}Parameters*, std::size_t,"
            ).replace("std::uint64_t, float*, float*", "std::uint64_t, "
                      "::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, "
                      "float*, float*, float*, float*")
            for suffix in ("cuh", "cu"):
                destination = output_root / f"{spec.unit_path}.{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                template = TEMPLATE_DIR / f"pricing/rough/volterra_fft/product_price_delta.{suffix}.tpl"
                _write_generated(destination, template.read_text().format(**values))
                generated.append(destination)
            continue
        if spec.pricing.engine == "equity_n_factor":
            binding = spec.pricing.manifest_binding
            values = rough_values(binding, spec.pricing.model,
                                  MODEL_BY_NAME[spec.pricing.model].display, "n_factor")
            values["explicit_instantiations"] = values["explicit_instantiations"].replace(
                f"launch_{spec.pricing.model}_{binding.product}_cuda",
                f"launch_{spec.pricing.model}_{binding.product}_price_delta_cuda"
            ).replace("const ModelParameters*, std::size_t,",
                      "const ModelParameters*, const ModelParameters*, std::size_t,").replace(
                "    float*, float*", "    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration,\n"
                "    float*, float*, float*, float*")
            for suffix in ("cuh", "cu"):
                destination = output_root / f"{spec.unit_path}.{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                template = TEMPLATE_DIR / f"pricing/rough/markovian_n_factor/product_price_delta.{suffix}.tpl"
                _write_generated(destination, template.read_text().format(**values))
                generated.append(destination)
            continue
        if spec.pricing.engine in {"equity_lsm_exact", "equity_lsm_fixed"}:
            model = spec.pricing.model
            exact = spec.pricing.engine == "equity_lsm_exact"
            schedule = "ExactTransition" if exact else "FixedStep"
            continuation = (
                f"product::SpotAndScaledStateContinuationState<{model}::DynamicsPolicy, "
                f"&{model}::State::variance, &{model}::ModelParameters::theta>"
                if model in {"heston", "bates"} else
                f"product::SpotAndScaledStateContinuationState<{model}::DynamicsPolicy, "
                f"&{model}::State::volatility, &{model}::ModelParameters::long_run_volatility>"
                if model == "schobel_zhu" else
                f"product::SpotLogMoneynessContinuationState<{model}::DynamicsPolicy>"
            )
            frozen_path = (
                "::ai_factory::workbench::equity::price_delta::MultiplicativeFrozenExercise"
                if spec.path_strategy == "multiplicative" else
                "::ai_factory::workbench::equity::price_delta::CoupledFrozenExercise<"
                f"simulation::{schedule}MaturityAlignedExerciseSchedule<{model}::PriceDeltaDynamics>, "
                f"::ai_factory::workbench::equity::price_delta::CoupledSpotPaths<{model}::PriceDeltaDynamics>>"
            )
            values = {
                "model": model, "schedule": schedule, "continuation": continuation,
                "regression_refinement": "normal_residual" if model == "kou" else "none",
                "frozen_path": frozen_path,
                "dynamics_header": ("price_delta_dynamics_impl.cuh" if spec.path_strategy == "coupled"
                                    else "dynamics_impl.cuh"),
                "time_declaration": "float day_fraction" if exact else "float dt, std::uint32_t simulation_steps_per_day",
                "time_types": "float" if exact else "float, std::uint32_t",
                "time_configuration": ("simulation::ExactTransitionTimeConfiguration{day_fraction}" if exact
                                       else "simulation::FixedStepTimeConfiguration{dt, simulation_steps_per_day}"),
            }
            for suffix in ("cuh", "cu"):
                destination = output_root / f"{spec.unit_path}.{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_generated(destination, _render_dollar_template(
                    f"pricing/longstaff_schwartz/equity/american_option_price_delta.{suffix}.tpl", values
                ))
                generated.append(destination)
            continue
        if spec.path_strategy == "closed_form_bump":
            for suffix in ("cuh", "cu"):
                destination = output_root / f"{spec.unit_path}.{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_generated(destination, _render_dollar_template(
                    f"pricing/closed_form/black_scholes/product_price_delta.{suffix}.tpl",
                    closed_form_delta_values(spec)))
                generated.append(destination)
            continue
        binding = spec.pricing.manifest_binding
        if not isinstance(binding, Binding):
            raise TypeError(f"missing price-delta Binding: {spec.unit_path}")
        product = next(p for p in ROUGH_PRODUCT_BINDINGS if p.product == binding.product)
        path_policy = (
            f"equity::price_delta::MultiplicativeSpotPath<{binding.model}::DynamicsPolicy>"
            if spec.path_strategy == "multiplicative"
            else f"equity::price_delta::CoupledSpotPaths<{binding.model}::PriceDeltaDynamics>"
        )
        values = {
            "model": binding.model,
            "product": binding.product,
            "product_type": binding.product_type,
            "schedule": binding.schedule,
            "calendar_arguments": ", 2U" if binding.schedule.endswith("CalendarSchedule") else "",
            "path_policy": path_policy,
            "product_path_policy": product.path_policy,
            "side_declaration": "template<OptionSide Side>" if binding.sided else "",
            "side_argument": "<Side>" if binding.sided else "",
            "probe_policy": "PricingPolicy<OptionSide::call>" if binding.sided else "PricingPolicy",
            "diagnostic_variant": "option_side_name(Side)" if binding.sided else '"default"',
            "time_declaration": "float dt, std::uint32_t simulation_steps_per_day" if binding.time_kind == "fixed" else "float day_fraction",
            "time_configuration": "{dt, simulation_steps_per_day}" if binding.time_kind == "fixed" else "{day_fraction}",
            "time_types": "float, std::uint32_t" if binding.time_kind == "fixed" else "float",
            "dynamics_header": (
                "price_delta_dynamics_impl.cuh" if spec.path_strategy == "coupled"
                else "dynamics_impl.cuh"
            ),
        }
        values["instantiations"] = "\n".join(
            _render_dollar_template("pricing/markovian/product_price_delta_instantiation.cu.tpl",
                                   {**values, "side": side})
            for side in (("call", "put") if binding.sided else ())
        )
        for suffix in ("cuh", "cu"):
            destination = output_root / f"{spec.unit_path}.{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_generated(destination, _render_dollar_template(
                f"pricing/markovian/product_price_delta_binding.{suffix}.tpl", values
            ))
            generated.append(destination)
    return generated


def generate_price_delta_recipes(output_root: Path) -> list[Path]:
    generated = []
    variants = {variant.name: variant for variant in PRICE_VARIANTS}
    for dataset in PRICE_DELTA_DATASET_SPECS:
        source = PRICE_DELTA_SOURCE_BY_GENERATOR[dataset.generator_path]
        spec = next(s for s in PRICE_DELTA_BINDING_SPECS
                    if (s.pricing.model, s.pricing.product) == (dataset.model, dataset.product))
        model = MODEL_BY_NAME[dataset.model]
        lsm = dataset.engine in {"equity_lsm_exact", "equity_lsm_fixed"}
        stochastic = dataset.engine != "equity_closed_form"
        variant = variants.get(dataset.variant)
        side = ("call" if dataset.variant == "american_calls" else "put") if lsm else variant.side
        product_loader = "product::load_american_options" if lsm else f"product::{variant.product_loader}"
        if variant and variant.side_aware_loader:
            product_loader = _render_cpp_fragment_template(
                "catalog/pricing/side_aware_product_loader_expression.cuh.tpl",
                {"product_loader": variant.product_loader, "side": side}).rstrip()
        product_id = ("american_options_01" if lsm else variant.product_dataset_id)
        prepared = dataset.engine == "equity_n_factor"
        fft = dataset.engine == "equity_volterra_fft"
        fixed = True if prepared or fft else dataset.engine == "equity_lsm_fixed" if lsm else (
            spec.pricing.manifest_binding.time_kind == "fixed" if stochastic else
            dataset.product == "geometric_asian_option")
        time_args = "1.0f / 504.0f, 2U" if fixed else "1.0f / 252.0f"
        args = ["host_models", "device_models", "model_count"]
        if stochastic or fixed:
            args.append("host_products")
        construction = (
            "CartesianProduct" if dataset.construction == "cartesian" else "Aligned"
        )
        args += ["device_products", "product_count", f"PriceConstruction::{construction}", "context.results"]
        if not lsm:
            args += ["context.offset", "context.count"]
        if stochastic:
            args += ["context.paths"]
        args += [time_args, "context.threads", "context.blocks"]
        if stochastic:
            args.append("context.seed")
        args += ["context.bump", "prices"]
        if stochastic:
            args.append("price_errors")
        args.append("deltas")
        if stochastic:
            args.append("delta_errors")
        values = {
            "model": dataset.model, "product": dataset.product,
            "model_input": f"datasets/{model.source_prefix}/parameters/{model.parameter_dataset_id}.json",
            "product_input": f"datasets/product/{dataset.product}/{product_id}.json",
            "dataset": dataset.dataset_path, "catalog": dataset.generation_yaml_path,
            "url": dataset.url, "source_recipe": source.recipe_yaml_path,
            "method": "frozen_central_exercise_dates_crn" if lsm else "centered_crn" if stochastic else "centered_closed_form",
            "stochastic": str(stochastic).lower(), "lsm": str(lsm).lower(),
            "steps_per_day": "2U" if fixed else "0U",
            "family": pricing_launch_family(spec.pricing),
            "seed": str(resolve_rng_domain(dataset).seed("dynamics")) if stochastic else "0",
            "product_loader": product_loader,
            "side": f"<OptionSide::{side}>" if side else "",
            "arguments": ",\n                    ".join(args),
            "construction": construction,
            "construction_label": (
                "Cartesian-product" if dataset.construction == "cartesian" else "aligned"
            ),
            "construction_prefix": (
                "Cartesian-product " if dataset.construction == "cartesian" else ""
            ),
            "construction_argument": (
                ", PriceConstruction::CartesianProduct"
                if dataset.construction == "cartesian" else ""
            ),
        }
        destination = output_root / dataset.generator_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        template = dataset.template
        if fft:
            values["schedule"] = rough_schedules(spec.pricing.manifest_binding)[0]
            values["product_policy"] = "product::" + spec.pricing.manifest_binding.path_policy + (f"<OptionSide::{side}>" if side else "")
            values["numerical_method"] = SAMPLE_MODEL_BY_NAME[dataset.model].pricing_numerical_method
        if prepared:
            template = "catalog/pricing/price_delta/prepared_generator.cpp.tpl"
            values["template_arguments"] = f"OptionSide::{side}, 7U" if side else "7U"
            values["numerical_method"] = SAMPLE_MODEL_BY_NAME[dataset.model].pricing_numerical_method
        _write_generated(destination, _render_dollar_template(template, values))
        generated.append(destination)
        recipe = {
            "schema_version": 1, "kind": "price_delta", "dataset_id": dataset.dataset_id,
            "generator": "generator.cpp", "model_input": values["model_input"],
            "product_input": values["product_input"],
            "output": {"path": dataset.dataset_path, "format": "json"},
            "generation_output": dataset.generation_yaml_path,
            "construction": dataset.construction,
            "paths_per_price": 1048576 if stochastic else 0,
            "launch_profile": "inherited price profile; inspect compiled plan; not delta-tuned",
            "sensitivity": {"parameter": "spot", "method": values["method"],
                            "relative_full_width": .01, "source_price_recipe": source.recipe_yaml_path},
            "dynamics_seed": int(values["seed"]),
            "time_grid": {"steps_per_year": 504, "simulation_steps_per_day": 2, "delta_t": "1 / 504"} if fixed else None,
        }
        recipe_path = destination.with_name("recipe.yaml")
        if prepared:
            recipe["preparation"] = {"method": values["numerical_method"], "factor_count": 7,
                                     "approximation_horizon_rule": "maximum product maturity in years",
                                     "coefficient_precision": "host FP64, device FP32"}
        if fft:
            recipe["preparation"] = {"method": values["numerical_method"], "shared_convolution": True}
        _write_generated(recipe_path, json.dumps(recipe, indent=2) + "\n")
        generated.append(recipe_path)
    return generated


def rough_schedules(binding: RoughProductBinding) -> tuple[str, str]:
    schedules = {
        "terminal": (
            "volterra::TerminalHybridSchedule",
            "simulation::FixedStepTerminalSchedule<"
            "DynamicsPolicy<FactorCount>>",
        ),
        "dense": (
            "volterra::DenseHybridSchedule",
            "simulation::FixedStepDenseSchedule<"
            "DynamicsPolicy<FactorCount>>",
        ),
        "regular": (
            "volterra::RegularHybridSchedule",
            "simulation::FixedStepRegularSchedule<"
            "DynamicsPolicy<FactorCount>>",
        ),
        "calendar_2": (
            "volterra::CalendarHybridSchedule<2U>",
            "simulation::FixedStepCalendarSchedule<"
            "DynamicsPolicy<FactorCount>, 2U>",
        ),
    }
    try:
        return schedules[binding.schedule_kind]
    except KeyError as error:
        raise ValueError(
            f"Unsupported rough schedule kind: {binding.schedule_kind}"
        ) from error


def volterra_instantiation(
    model: str,
    binding: RoughProductBinding,
    side: str,
) -> str:
    return f"""template void launch_{model}_{binding.product}_cuda<
    OptionSide::{side}
>(
    const ModelParameters*, std::size_t,
    const product::{binding.product_type}Parameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t,
    float, float, std::size_t, std::size_t,
    void*, std::size_t, std::uint64_t, float*, float*
);
"""


def n_factor_instantiation(
    model: str,
    binding: RoughProductBinding,
    factor_count: int,
    side: str | None,
) -> str:
    arguments = f"OptionSide::{side}, {factor_count}U" if side else f"{factor_count}U"
    return f"""template void launch_{model}_{binding.product}_cuda<
    {arguments}
>(
    const ModelParameters*, std::size_t,
    const PreparedDynamics<{factor_count}U>*, std::size_t,
    const product::{binding.product_type}Parameters*,
    const product::{binding.product_type}Parameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, std::uint64_t,
    float*, float*
);
"""


def rough_values(
    binding: RoughProductBinding,
    model: str,
    model_display: str,
    backend: str,
) -> dict[str, str]:
    volterra_schedule, n_factor_schedule = rough_schedules(binding)
    if backend == "volterra":
        template_declaration = "template<OptionSide Side>\n" if binding.sided else ""
        explicit_instantiations = ""
        if binding.sided:
            explicit_instantiations = "\n".join(
                volterra_instantiation(model, binding, side)
                for side in ("call", "put")
            )
    elif backend == "n_factor":
        template_declaration = (
            "template<OptionSide Side, std::size_t FactorCount>\n"
            if binding.sided
            else "template<std::size_t FactorCount>\n"
        )
        sides = ("call", "put") if binding.sided else (None,)
        explicit_instantiations = "\n".join(
                n_factor_instantiation(model, binding, factor_count, side)
            for factor_count in (2, 3, 7)
            for side in sides
        )
    else:
        raise ValueError(f"Unsupported rough backend: {backend}")

    return {
        **binding.__dict__,
        "model": model,
        "model_display": model_display,
        "product_comment": binding.product.replace("_", "-"),
        "header_product_comment": binding.product.replace(
            "_", "-"
        ).capitalize(),
        "operation_product": binding.product.replace("_", " "),
        "template_declaration": template_declaration,
        "product_policy_expression": (
            f"product::{binding.path_policy}<Side>"
            if binding.sided
            else f"product::{binding.path_policy}"
        ),
        "diagnostic_variant": (
            "option_side_name(Side)" if binding.sided else '"default"'
        ),
        "volterra_schedule": volterra_schedule,
        "n_factor_schedule": n_factor_schedule,
        "explicit_instantiations": explicit_instantiations,
    }


def generate_rough(output_root: Path) -> list[Path]:
    templates = {
        "volterra_header": (
            TEMPLATE_DIR / "pricing" / "rough" / "volterra_fft"
            / "product_binding.cuh.tpl"
        ).read_text(),
        "volterra_source": (
            TEMPLATE_DIR / "pricing" / "rough" / "volterra_fft"
            / "product_binding.cu.tpl"
        ).read_text(),
        "n_factor_header": (
            TEMPLATE_DIR / "pricing" / "rough" / "markovian_n_factor"
            / "product_binding.cuh.tpl"
        ).read_text(),
        "n_factor_source": (
            TEMPLATE_DIR / "pricing" / "rough" / "markovian_n_factor"
            / "product_binding.cu.tpl"
        ).read_text(),
    }
    generated: list[Path] = []
    backends = {
        "equity_volterra_fft": "volterra",
        "equity_n_factor": "n_factor",
    }
    for spec in GENERATED_PRODUCT_BINDING_SPECS:
        if spec.engine not in backends:
            continue
        binding = spec.manifest_binding
        if not isinstance(binding, RoughProductBinding):
            raise TypeError(f"missing rough product binding: {spec.unit_path}")
        backend = backends[spec.engine]
        values = rough_values(
            binding,
            spec.model,
            MODEL_BY_NAME[spec.model].display,
            backend,
        )
        destination = (output_root / spec.unit_path).parent
        destination.mkdir(parents=True, exist_ok=True)
        header = destination / f"{binding.product}.cuh"
        source = destination / f"{binding.product}.cu"
        _write_generated(header, templates[f"{backend}_header"].format(**values))
        _write_generated(source, templates[f"{backend}_source"].format(**values))
        generated.extend((header, source))
    return generated


def _fixed_income_template_values(
    capability,
    model_by_name=MODEL_BY_NAME,
    curve_by_name=CURVE_BY_NAME,
) -> dict[str, str]:
    model = model_by_name[capability.model]
    values = {
        "model": capability.model,
        "model_display": model.display,
        "model_alias": model.renderer_alias or model.name,
        "curve": capability.curve or "",
        "curve_display": "",
        "curve_type": "",
        "forward_dynamics": capability.terminal_forward_dynamics or "",
    }
    if capability.curve is not None:
        curve = curve_by_name[capability.curve]
        values["curve_display"] = curve.display
        values["curve_type"] = curve.cpp_type
    return values


def generate_fixed_income_bindings(output_root: Path) -> list[Path]:
    """Generate the fixed-income compositions whose engines have explicit templates."""
    generated: list[Path] = []
    for spec in GENERATED_PRODUCT_BINDING_SPECS:
        if spec.engine not in {"fixed_income_closed_form", "fixed_income_monte_carlo", "fixed_income_lsm"}:
            continue
        if spec.template_family is None:
            raise ValueError(
                f"generated fixed-income binding lacks a template: "
                f"{spec.unit_path}"
            )
        values = _fixed_income_template_values(spec)
        destination_directory = (output_root / spec.unit_path).parent
        destination_directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("cuh", "cu"):
            template = f"{spec.template_family}/{spec.product}.{suffix}.tpl"
            destination = destination_directory / f"{spec.product}.{suffix}"
            _write_generated(destination, _render_dollar_template(template, values))
            generated.append(destination)
    return generated


def pricing_identity_expression(dataset) -> str:
    binding = next(binding for binding in PRODUCT_BINDING_SPECS
                   if (binding.model, binding.curve, binding.product)
                   == (dataset.model, dataset.curve, dataset.product))
    namespace = "::ai_factory::workbench::offline::cuda_tuning::"
    fields = ", ".join(json.dumps(value or "") for value in
                       (dataset.model, dataset.product, dataset.curve))
    return f"{namespace}PricingIdentity{{{namespace}PricingFamily::{pricing_launch_family(binding)}, {fields}}}"


def _fixed_income_recipe_values(dataset) -> dict[str, str]:
    capability = next(
        candidate for candidate in FIXED_INCOME_CAPABILITIES
        if candidate.model == dataset.model
        and candidate.curve == dataset.curve
    )
    values = _fixed_income_template_values(capability)
    values.update({
        "database_id": dataset.dataset_id,
        "price_dataset_path": dataset.dataset_path,
        "catalog_path": dataset.generation_yaml_path,
        "url": dataset.url,
        "construction": (
            "CartesianProduct"
            if dataset.construction == "cartesian" else "Aligned"
        ),
        "launch_identity": pricing_identity_expression(dataset),
        "variant": dataset.variant or "",
        "payoff_name": {
            "caplets": "caplet",
            "floorlets": "floorlet",
            "zero_coupon_bond_calls": "zero-coupon bond call",
            "zero_coupon_bond_puts": "zero-coupon bond put",
        }.get(dataset.variant or "", ""),
        "pricing_formula": {
            "caplets": "zero-coupon bond put",
            "floorlets": "zero-coupon bond call",
        }.get(dataset.variant or "", ""),
        "option_side": {
            "caplets": "call",
            "floorlets": "put",
            "zero_coupon_bond_calls": "call",
            "zero_coupon_bond_puts": "put",
        }.get(dataset.variant or "", ""),
        "swaption_side": (
            "payer" if dataset.variant in {"european_payer_swaptions", "bermudan_payer_swaptions"}
            else "receiver"
        ),
        "bond_option_side_plural": (
            "puts" if dataset.variant == "european_payer_swaptions"
            else "calls"
        ),
    })
    bermudan_methods = {
        "cir": (
            "Exact CIR terminal-forward transitions + Longstaff-Schwartz",
            "Hermite degree 3", "standardized short-rate factor",
        ),
        "cir_plus_plus": (
            "Exact fitted CIR terminal-forward transitions + Longstaff-Schwartz",
            "Hermite degree 3", "standardized unshifted CIR factor",
        ),
        "g2": (
            "Exact two-factor Gaussian joint transition + Longstaff-Schwartz",
            "two-factor Hermite degree 2", "two standardized rate factors",
        ),
        "g2_plus_plus": (
            "Exact fitted two-factor Gaussian joint transition + Longstaff-Schwartz",
            "two-factor Hermite degree 2", "two standardized rate factors",
        ),
        "hull_white": (
            "Exact fitted Gaussian joint transition + Longstaff-Schwartz",
            "Hermite degree 3", "standardized centered short-rate factor",
        ),
        "ornstein_uhlenbeck": (
            "Exact Gaussian joint transition + Longstaff-Schwartz",
            "Hermite degree 3", "standardized short-rate factor",
        ),
        "vasicek": (
            "Exact Gaussian joint transition + Longstaff-Schwartz",
            "Hermite degree 3", "standardized short-rate factor",
        ),
    }
    if dataset.product == "bermudan_swaption":
        method, basis, state = bermudan_methods[dataset.model]
        values.update({
            "bermudan_numerical_method": method,
            "bermudan_regression_basis": basis,
            "bermudan_state_variables": state,
            "bermudan_configuration_overrides": (
                '    configuration.pricing_measure = "last_exercise_bond_forward";\n'
                '    configuration.regression_target = "next policy cashflow in '
                'P(0,T*) / P(t,T*) units";\n'
                if dataset.model == "cir" else ""
            ),
        })
    if dataset.engine in {"fixed_income_monte_carlo", "fixed_income_lsm"}:
        values["dynamics_seed"] = str(resolve_rng_domain(dataset).seed("dynamics"))
    return values


def generate_fixed_income_catalog_recipes(output_root: Path) -> list[Path]:
    generated: list[Path] = []
    datasets = (
        dataset for dataset in AVAILABLE_DATASET_SPECS
        if dataset.engine in {"fixed_income_closed_form", "fixed_income_monte_carlo", "fixed_income_lsm"}
        and dataset.owner == "generated"
    )
    for dataset in datasets:
        if dataset.template is None:
            raise ValueError(
                f"Generated fixed-income dataset lacks a template: "
                f"{dataset.generator_path}"
            )
        destination = output_root / dataset.generator_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = _render_dollar_template(
            dataset.template,
            _fixed_income_recipe_values(dataset),
        )
        if dataset.construction == "cartesian":
            rendered = rendered.replace(
                dataset.dataset_id.removesuffix("_cartesian"),
                dataset.dataset_id,
            )
        _write_generated(destination, rendered)
        generated.append(destination)
    return generated


MONTE_CARLO_PATHS_PER_PRICE = "::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice"


def price_recipe_url(model, variant, database_id: str) -> str:
    source_prefix = MODEL_BY_NAME[model.name].source_prefix
    return (
        "https://datasets.ai-factory.example/v1/"
        f"{source_prefix}/prices/{variant.name}/{database_id}.json"
    )


def markovian_time_values(model: str, product: str) -> dict[str, str]:
    binding = next(
        candidate
        for candidate in BINDINGS
        if candidate.model == model and candidate.product == product
    )
    if binding.time_kind == "exact":
        return {
            "time_constants": "    constexpr float day_fraction = 1.0f / 252.0f;\n",
            "time_arguments": "                        day_fraction,\n",
            "delta_t_description": "exact transition dates",
            "execution_metadata": "nlohmann::ordered_json::object()",
        }
    return {
        "time_constants": (
            "    constexpr float dt = 1.0f / 504.0f;\n"
            "    constexpr std::uint32_t simulation_steps_per_day = 2U;\n"
        ),
        "time_arguments": (
            "                        dt,\n"
            "                        simulation_steps_per_day,\n"
        ),
        "delta_t_description": "1 / 504",
        "execution_metadata": (
            "nlohmann::ordered_json{"
            "{\"simulation_steps_per_day\", simulation_steps_per_day}"
            "}"
        ),
    }


def generate_catalog_recipes(output_root: Path) -> list[Path]:
    templates = {
        "markovian": (
            TEMPLATE_DIR / "catalog" / "pricing" / "markovian"
            / "generator.cpp.tpl"
        ).read_text(),
        "n_factor": (
            TEMPLATE_DIR / "catalog" / "pricing" / "rough"
            / "markovian_n_factor" / "generator.cpp.tpl"
        ).read_text(),
        "volterra": (
            TEMPLATE_DIR / "catalog" / "pricing" / "rough"
            / "volterra_fft" / "generator.cpp.tpl"
        ).read_text(),
        "closed_form": (
            TEMPLATE_DIR / "catalog" / "pricing"
            / "black_scholes_closed_form" / "generator.cpp.tpl"
        ).read_text(),
        "american": (
            TEMPLATE_DIR / "catalog" / "pricing"
            / "american_longstaff_schwartz" / "generator.cpp.tpl"
        ).read_text(),
    }
    generated: list[Path] = []
    model_recipes = {model.name: model for model in MODEL_RECIPE_SPECS}
    variants = {variant.name: variant for variant in PRICE_VARIANTS}
    template_keys = {
        "equity_markovian": "markovian",
        "equity_n_factor": "n_factor",
        "equity_volterra_fft": "volterra",
        "equity_closed_form": "closed_form",
    }
    ordinary_datasets = (
        dataset for dataset in AVAILABLE_DATASET_SPECS
        if dataset.dataset_kind == "prices"
        and dataset.owner == "generated"
        and dataset.engine in template_keys
    )
    for dataset in ordinary_datasets:
        model = model_recipes[dataset.model or ""]
        model_spec = MODEL_BY_NAME[model.name]
        variant = variants[dataset.variant or ""]
        database_id = dataset.dataset_id
        backend = template_keys[dataset.engine or ""]
        destination = output_root / dataset.generator_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        side_template = (
            f"<OptionSide::{variant.side}>" if variant.side else ""
        )
        product_loader_expression = f"product::{variant.product_loader}"
        if variant.side_aware_loader:
            product_loader_expression = _render_cpp_fragment_template(
                "catalog/pricing/side_aware_product_loader_expression.cuh.tpl",
                {
                    "product_loader": variant.product_loader,
                    "side": variant.side or "",
                },
            ).rstrip()
        values = {
            "model": model.name,
            "model_display": model.display,
            "product": variant.product,
            "product_loader_expression": product_loader_expression,
            "variant_comment": variant.name.replace("_", "-"),
            "side_template": side_template,
            "template_arguments": (
                f"OptionSide::{variant.side}, factor_count"
                if variant.side else "factor_count"
            ),
            "model_dataset_path": (
                f"datasets/{model_spec.source_prefix}/parameters/"
                f"{model_spec.parameter_dataset_id}.json"
            ),
            "product_dataset_path": (
                f"datasets/product/{variant.product}/"
                f"{variant.product_dataset_id}.json"
            ),
            "price_dataset_path": dataset.dataset_path,
            "catalog_path": dataset.generation_yaml_path,
            "url": dataset.url,
            "numerical_method": model.numerical_method,
            "monte_carlo_paths": MONTE_CARLO_PATHS_PER_PRICE,
            "launch_identity": pricing_identity_expression(dataset),
            "construction": (
                "CartesianProduct"
                if dataset.construction == "cartesian" else "Aligned"
            ),
            "seed": (
                str(resolve_rng_domain(dataset).seed("dynamics"))
                if dataset.engine != "equity_closed_form" else "0"
            ),
        }
        analytical_steps = variant.analytical_steps_per_day
        values["threads_per_block"] = (
            "::ai_factory::workbench::offline::cuda_tuning::pricing_profile("
            + values["launch_identity"] + ").threads_per_block"
        )
        values["analytical_profile_values"] = (
            f"1.0f / {252 * analytical_steps}.0f, "
            "::ai_factory::workbench::offline::cuda_tuning::kAnalyticalThreadsPerBlock, "
            f"{analytical_steps}U"
            if analytical_steps is not None
            else "1.0f / 252.0f, "
            "::ai_factory::workbench::offline::cuda_tuning::kAnalyticalThreadsPerBlock, 0U"
        )
        values["analytical_time_arguments"] = (
            "                        context.day_fraction,\n"
            "                        context.simulation_steps_per_day,"
            if analytical_steps is not None
            else "                        context.day_fraction,"
        )
        values["analytical_host_product_argument"] = (
            "                        products.data(),"
            if variant.product == "geometric_asian_option"
            else ""
        )
        if backend == "markovian":
            values.update(markovian_time_values(
                model.name, variant.product
            ))
        _write_generated(destination, templates[backend].format(**values))
        generated.append(destination)
    american_models = {model.model: model for model in AMERICAN_RECIPE_SPECS}
    american_datasets = (
        dataset for dataset in AVAILABLE_DATASET_SPECS
        if dataset.dataset_kind == "prices"
        and dataset.owner == "generated"
        and dataset.engine in {"equity_lsm_fixed", "equity_lsm_exact"}
    )
    for dataset in american_datasets:
        model = american_models[dataset.model or ""]
        side = "call" if dataset.variant == "american_calls" else "put"
        database_id = dataset.dataset_id
        destination = output_root / dataset.generator_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if model.time_kind == "fixed":
            time_values = {
                "time_constants": (
                    "    constexpr float dt = 1.0f / 504.0f;\n"
                    "    constexpr std::uint32_t simulation_steps_per_day = 2U;\n"
                ),
                "time_arguments": (
                    "                dt,\n"
                    "                simulation_steps_per_day,\n"
                ),
                "delta_t_description": "1 / 504",
                "time_discretization": (
                    "nlohmann::ordered_json{{\"simulation_steps_per_day\", "
                    "simulation_steps_per_day}}"
                ),
                "exact_exercise_dates": "false",
            }
        elif model.time_kind == "exact":
            time_values = {
                "time_constants": (
                    "    constexpr float day_fraction = 1.0f / 252.0f;\n"
                ),
                "time_arguments": "                day_fraction,\n",
                "delta_t_description": "",
                "time_discretization": "nlohmann::ordered_json::object()",
                "exact_exercise_dates": "true",
            }
        else:
            raise ValueError(
                f"Unsupported American time kind: {model.time_kind}"
            )

        def quoted(values: tuple[str, ...]) -> str:
            return "{" + ", ".join(
                f'\"{value}\"' for value in values
            ) + "}"

        values = {
            **model.__dict__,
            **time_values,
            "side": side,
            "database_id": database_id,
            "diagnostic_label": (
                f"{model.model.replace('_', '-')} American {side}"
            ),
            "basis_state": quoted(model.basis_state),
            "basis_normalization": quoted(model.basis_normalization),
            "basis_functions": quoted(model.basis_functions),
            "seed": str(resolve_rng_domain(dataset).seed("dynamics")),
            "launch_identity": pricing_identity_expression(dataset),
            "construction": (
                "CartesianProduct"
                if dataset.construction == "cartesian" else "Aligned"
            ),
        }
        _write_generated(destination, templates["american"].format(**values))
        generated.append(destination)
    return generated


def _cpp_string_literals(source: str) -> list[str]:
    groups = re.findall(
        r'"(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*',
        source,
    )
    return [
        "".join(
            json.loads(token)
            for token in re.findall(r'"(?:[^"\\]|\\.)*"', group)
        )
        for group in groups
    ]


def _price_recipe_metadata(dataset, source: str) -> dict:
    inputs = [
        value for value in _cpp_string_literals(source)
        if value.startswith("datasets/")
        and value.endswith(".json")
        and value != dataset.dataset_path
    ]
    roles: dict[str, str] = {}
    for value in inputs:
        if value.startswith("datasets/product/"):
            role = "product"
        elif value.startswith("datasets/curve/"):
            role = "curve"
        elif value.startswith("datasets/model/"):
            role = "model"
        else:
            raise ValueError(f"Unknown dataset input role: {value}")
        if role in roles:
            raise ValueError(f"Duplicate dataset input role: {dataset.generator_path}")
        roles[role] = value
    expected_roles = {"model", "product"} | ({"curve"} if dataset.curve else set())
    if roles.keys() != expected_roles:
        raise ValueError(
            f"Incomplete inputs for {dataset.generator_path}: {sorted(roles)}"
        )
    stochastic = dataset.engine not in {
        "equity_closed_form", "fixed_income_closed_form",
    }
    metadata: dict = {
        "schema_version": 1,
        "kind": "prices",
        "dataset_id": dataset.dataset_id,
        "generator": "generator.cpp",
        "inputs": roles,
        "output": {"path": dataset.dataset_path, "format": "json"},
        "generation_output": dataset.generation_yaml_path,
        "construction": dataset.construction,
        "numerical_method": {
            "engine": dataset.engine,
            "profile": dataset.numerical_profile,
        },
        "outputs": (
            ["price", "standard_error"] if stochastic else ["price"]
        ),
        "paths_per_price": 1_048_576 if stochastic else 0,
    }
    domain = resolve_rng_domain(dataset) if stochastic else None
    if domain is not None:
        metadata["seeds"] = {
            name: domain.seed(name) for name in domain.streams
        }
    denominators = [
        int(value) for value in re.findall(r"1\.0f\s*/\s*(\d+)\.0f", source)
    ]
    fixed_steps = [value for value in denominators if value > 252]
    if fixed_steps:
        steps_per_year = fixed_steps[0]
        metadata["time_grid"] = {
            "steps_per_year": steps_per_year,
            "simulation_steps_per_day": steps_per_year // 252,
            "delta_t": f"1 / {steps_per_year}",
        }
    return metadata


def generate_price_recipe_metadata(output_root: Path) -> list[Path]:
    generated: list[Path] = []
    for dataset in AVAILABLE_DATASET_SPECS:
        if dataset.dataset_kind != "prices" or dataset.owner != "generated":
            continue
        generator = output_root / dataset.generator_path
        metadata_path = output_root / dataset.recipe_yaml_path
        _write_generated(
            metadata_path,
            json.dumps(
                _price_recipe_metadata(dataset, generator.read_text()),
                indent=2,
            ) + "\n",
        )
        generated.append(metadata_path)
    return generated


def cmake_list(name: str, values: list[str]) -> str:
    body = "\n".join(f"    {value}" for value in values)
    return f"set({name}\n{body}\n)\n"


def cmake_manifest_text(
    model_specs=MODEL_SPECS,
    product_specs=PRODUCT_SPECS,
    curve_specs=CURVE_SPECS,
    dataset_specs=AVAILABLE_DATASET_SPECS,
    binding_specs=GENERATED_PRODUCT_BINDING_SPECS,
) -> str:
    """Render every CMake inventory projected from the typed manifest."""
    equity_models = [
        model.name for model in model_specs if model.asset_class == "equity"
    ]
    rough_models = [
        model.name for model in model_specs
        if model.asset_class == "equity"
        and model.family.startswith("rough_")
    ]
    fixed_income_models = [
        model.name for model in model_specs
        if model.asset_class == "fixed_income"
    ]
    equity_binding_specs = [
        spec for spec in binding_specs
        if spec.asset_class == "equity"
    ]
    products = sorted({spec.name for spec in product_specs})
    curves = sorted({spec.name for spec in curve_specs})
    regular_units = sorted({
        f"{spec.model}/product/{spec.product}"
        for spec in equity_binding_specs
        if spec.engine in {
            "equity_closed_form", "equity_markovian", "equity_n_factor",
        }
    })
    regular_units = sorted(set(regular_units) | {
        f"{spec.pricing.model}/product/{spec.pricing.product}_price_delta"
        for spec in PRICE_DELTA_BINDING_SPECS
        if (spec.pricing in equity_binding_specs and spec.pricing.engine != "equity_volterra_fft") or (
            spec.pricing.engine in {"equity_lsm_exact", "equity_lsm_fixed"}
            and spec.pricing.model in equity_models
        )
    })
    volterra_units = sorted({
        f"{spec.model}/product/{spec.product}"
        for spec in equity_binding_specs
        if spec.engine == "equity_volterra_fft"
    })
    regular_units = sorted(set(regular_units) | {
        f"{spec.pricing.model}/product/{spec.pricing.product}_price_gradients"
        for spec in PRICE_GRADIENT_BINDING_SPECS
        if spec.pricing in equity_binding_specs or (
            spec.pricing.engine in {"equity_lsm_exact", "equity_lsm_fixed"}
            and spec.pricing.model in equity_models
        )
    })
    volterra_units = sorted(set(volterra_units) | {
        f"{spec.pricing.model}/product/{spec.pricing.product}_price_delta"
        for spec in PRICE_DELTA_BINDING_SPECS
        if spec.pricing in equity_binding_specs and spec.pricing.engine == "equity_volterra_fft"
    })
    parameter_sources = sorted(
        dataset.generator_path for dataset in dataset_specs
        if dataset.dataset_kind.endswith("_parameters")
    )
    price_sources = sorted(
        dataset.generator_path for dataset in dataset_specs
        if dataset.dataset_kind in {"prices", "price_delta", "price_gradients"}
    )
    sample_sources = sorted(
        dataset.generator_path for dataset in dataset_specs
        if dataset.dataset_kind == "samples"
    )
    mathdx_sources = sorted(
        dataset.generator_path for dataset in dataset_specs
        if dataset.condition == "AI_FACTORY_MATHDX_ROOT"
    )
    return (
        "# Generated by tools/codegen/pricing_bindings/generate.py.\n"
        "# Do not edit: change the typed manifests and regenerate.\n\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_MODELS",
            equity_models,
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_FIXED_INCOME_MODELS",
            fixed_income_models,
        )
        + "\n"
        + cmake_list("AI_FACTORY_GENERATED_ROUGH_MODELS", rough_models)
        + "\n"
        + cmake_list("AI_FACTORY_GENERATED_CURVES", curves)
        + "\n"
        + cmake_list("AI_FACTORY_GENERATED_PRODUCTS", products)
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_REGULAR_UNITS", regular_units
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_VOLTERRA_UNITS", volterra_units
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_EARLY_EXERCISE_UNITS",
            list(EQUITY_EARLY_EXERCISE_UNITS),
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_SAMPLE_UNITS",
            list(EQUITY_SAMPLE_UNITS),
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_MATHDX_SAMPLE_UNITS",
            list(EQUITY_MATHDX_SAMPLE_UNITS),
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_FIXED_INCOME_UNITS",
            list(FIXED_INCOME_UNITS),
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_MANIFEST_PARAMETER_GENERATOR_SOURCES",
            parameter_sources,
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_MANIFEST_PRICE_GENERATOR_SOURCES", price_sources
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_MANIFEST_SAMPLE_GENERATOR_SOURCES", sample_sources
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_MANIFEST_MATHDX_GENERATOR_SOURCES", mathdx_sources
        )
    )


def generate_cmake_manifest(output_root: Path) -> list[Path]:
    """Emit the exact CMake registration matrix from the typed manifest."""
    destination = (
        output_root / "cmake" / "generated" / "CapabilityManifest.cmake"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_generated(destination, cmake_manifest_text())
    return [destination]


def codegen_source_fingerprint() -> str:
    sources = [
        SCRIPT_DIR / "manifest.py",
        SCRIPT_DIR / "capability_manifest.py",
        SCRIPT_DIR / "sample_manifest.py",
        SCRIPT_DIR / "generate.py",
        *sorted((SCRIPT_DIR / "price_gradients").glob("*.py")),
        *sorted(TEMPLATE_DIR.rglob("*.tpl")),
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(SCRIPT_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generate_provenance_manifest(
    output_root: Path,
    generated: list[Path],
) -> list[Path]:
    destination = (
        output_root / "cmake" / "generated"
        / "PricingCapabilityManifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_paths = sorted(generated, key=lambda path: path.relative_to(
        output_root
    ).as_posix())
    output_digest = hashlib.sha256()
    for path in output_paths:
        output_digest.update(path.relative_to(output_root).as_posix().encode())
        output_digest.update(b"\0")
        output_digest.update(path.read_bytes())
        output_digest.update(b"\0")
    _write_generated(destination, json.dumps({
        "schema_version": SCHEMA_VERSION,
        "source_sha256": codegen_source_fingerprint(),
        "counts": {
            "engines": len(ENGINE_SPECS),
            "models": len(MODEL_SPECS),
            "products": len(PRODUCT_SPECS),
            "available_datasets": len(AVAILABLE_DATASET_SPECS),
            "deferred_datasets": len(DEFERRED_DATASET_SPECS),
            "bounded_exceptions": len(CAPABILITY_EXCEPTIONS),
            "generated_outputs": len(output_paths),
        },
        "outputs_sha256": output_digest.hexdigest(),
        "price_delta_bindings": {
            spec.unit_path: {"pricing": spec.pricing.unit_path,
                             "identity": "/".join(filter(None, (spec.pricing.model, spec.pricing.curve, spec.pricing.product))),
                             "path_strategy": spec.path_strategy,
                             "qualification": "bounded_checks; bias_and_performance_not_certified"}
            for spec in PRICE_DELTA_BINDING_SPECS
        },
        "price_gradient_bindings": {
            spec.unit_path: {"identity": f"{spec.pricing.model}/{spec.pricing.product}",
                "maximum_sensitivities": spec.maximum_sensitivities,
                "qualification": "bounded_checks; bias_and_performance_not_certified"}
            for spec in PRICE_GRADIENT_BINDING_SPECS
        },
        "pricing_launch_families": {
            "/".join(filter(None, (binding.model, binding.curve, binding.product))):
                pricing_launch_family(binding)
            for binding in PRODUCT_BINDING_SPECS
        },
    }, indent=2) + "\n")
    return [destination]


def _relative_files(root: Path, pattern: str) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.glob(pattern)
        if path.is_file()
    }


def _repository_inventory_diagnostics(reference_root: Path) -> list[str]:
    diagnostics: list[str] = []
    inventory = (
        (
            "product binding",
            set(DECLARED_PRODUCT_BINDING_PATHS) | set(GENERATED_PRICE_DELTA_BINDING_PATHS)
            | set(GENERATED_PRICE_GRADIENT_BINDING_PATHS)
            | set(GENERATED_CLOSED_FORM_POLICY_PATHS),
            _relative_files(reference_root, "src/model/**/product/**/*.cu")
            | _relative_files(reference_root, "src/model/**/product/**/*.cuh"),
        ),
        (
            "catalog recipe",
            {dataset.generator_path for dataset in AVAILABLE_DATASET_SPECS},
            _relative_files(reference_root, "catalog/**/generator.cpp"),
        ),
        (
            "sample binding",
            {
                f"src/{model.source_prefix}/sample.{suffix}"
                for model in MODEL_SPECS
                if model.sample_binding_status == "available"
                for suffix in ("cuh", "cu")
            },
            _relative_files(reference_root, "src/model/**/sample.cu")
            | _relative_files(reference_root, "src/model/**/sample.cuh"),
        ),
        (
            "sample helper",
            {
                f"tools/sampling/generated/{model.name}_sample_generation.cuh"
                for model in MODEL_SPECS
                if model.sample_binding_status == "available"
            },
            _relative_files(
                reference_root,
                "tools/sampling/generated/*_sample_generation.cuh",
            ),
        ),
    )
    for artifact_kind, expected, actual in inventory:
        diagnostics.extend(
            f"CODEGEN_MISSING [{artifact_kind}] {path}"
            for path in sorted(expected - actual)
        )
        diagnostics.extend(
            f"CODEGEN_EXTRA [{artifact_kind}] {path}"
            for path in sorted(actual - expected)
        )
    return diagnostics


def _expected_generated_paths() -> set[str]:
    paths = set(GENERATED_PRODUCT_BINDING_PATHS)
    paths.update(GENERATED_PRICE_DELTA_BINDING_PATHS)
    paths.update(GENERATED_PRICE_GRADIENT_BINDING_PATHS)
    paths.update(GENERATED_CLOSED_FORM_POLICY_PATHS)
    paths.update(str(Path(dataset.generator_path).with_name("recipe.yaml"))
                 for dataset in PRICE_DELTA_DATASET_SPECS)
    paths.update(str(Path(dataset.generator_path).with_name("recipe.yaml"))
                 for dataset in PRICE_GRADIENT_DATASET_SPECS)
    paths.update(dataset.recipe_yaml_path for dataset in AVAILABLE_DATASET_SPECS
                 if dataset.dataset_kind in {"prices", "samples"})
    paths.update(
        dataset.generator_path for dataset in AVAILABLE_DATASET_SPECS
        if dataset.owner == "generated"
    )
    for model in MODEL_SPECS:
        if model.sample_binding_status != "available":
            continue
        paths.update({
            f"src/{model.source_prefix}/sample.cuh",
            f"src/{model.source_prefix}/sample.cu",
            f"tools/sampling/generated/{model.name}_sample_generation.cuh",
        })
    paths.update({
        "cmake/generated/CapabilityManifest.cmake",
        "cmake/generated/PricingCapabilityManifest.json",
    })
    return paths


def compare(
    generated: list[Path],
    output_root: Path,
    reference_root: Path,
    verify_inventory: bool = False,
) -> int:
    mismatch_count = 0
    diagnostics = (
        _repository_inventory_diagnostics(reference_root)
        if verify_inventory else []
    )
    if verify_inventory:
        generated_paths = {
            path.relative_to(output_root).as_posix() for path in generated
        }
        expected_generated_paths = _expected_generated_paths()
        diagnostics.extend(
            f"CODEGEN_RENDERER_MISSING {path}"
            for path in sorted(expected_generated_paths - generated_paths)
        )
        diagnostics.extend(
            f"CODEGEN_RENDERER_UNDECLARED {path}"
            for path in sorted(generated_paths - expected_generated_paths)
        )
    if diagnostics:
        print("\n".join(diagnostics))
        mismatch_count += len(diagnostics)
    for generated_path in generated:
        relative_path = generated_path.relative_to(output_root)
        reference_path = reference_root / relative_path
        if not reference_path.is_file():
            mismatch_count += 1
            print(f"CODEGEN_MISSING [generated output] {relative_path}")
            continue
        reference = reference_path.read_text()
        current = generated_path.read_text()
        if current == reference:
            continue
        mismatch_count += 1
        print(f"CODEGEN_MISMATCH {relative_path}")
        print("".join(difflib.unified_diff(
            reference.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=str(reference_path),
            tofile=str(generated_path),
        )))
    return mismatch_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ai_factory-pricing-bindings"),
    )
    parser.add_argument(
        "--family",
        choices=(
            "rough", "markovian", "prototype", "fixed_income", "catalog",
            "samples", "all"
        ),
        default="rough",
    )
    parser.add_argument("--compare-root", type=Path)
    arguments = parser.parse_args()

    generated: list[Path] = []
    if arguments.family in ("markovian", "prototype", "all"):
        generated.extend(generate_markovian(arguments.output))
        generated.extend(generate_price_delta_bindings(arguments.output))
        generated.extend(render_price_gradient_bindings(arguments.output, PRICE_GRADIENT_BINDING_SPECS,
            TEMPLATE_DIR, _write_generated))
    if arguments.family in ("rough", "all"):
        generated.extend(generate_rough(arguments.output))
    if arguments.family in ("fixed_income", "all"):
        generated.extend(generate_fixed_income_bindings(arguments.output))
    if arguments.family in ("catalog", "all"):
        generated.extend(generate_price_delta_recipes(arguments.output))
        generated.extend(render_price_gradient_recipes(arguments.output, PRICE_GRADIENT_DATASET_SPECS,
            PRICE_GRADIENT_SOURCE_BY_GENERATOR, MODEL_BY_NAME, resolve_rng_domain, TEMPLATE_DIR, _write_generated))
        generated.extend(generate_catalog_recipes(arguments.output))
        generated.extend(generate_fixed_income_catalog_recipes(
            arguments.output
        ))
        generated.extend(generate_price_recipe_metadata(arguments.output))
    if arguments.family in ("samples", "all"):
        generated.extend(generate_samples(arguments.output))
    if arguments.family == "all":
        generated.extend(generate_cmake_manifest(arguments.output))
        generated.extend(generate_provenance_manifest(
            arguments.output, generated
        ))
    for path in generated:
        print(path)
    if arguments.compare_root is None:
        return 0
    return 1 if compare(
        generated,
        arguments.output,
        arguments.compare_root,
        verify_inventory=arguments.family == "all",
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
