#!/usr/bin/env python3
"""Generate thin model-product pricing bindings into an output folder."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
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
    DEFERRED_DATASET_SPECS,
    ENGINE_SPECS,
    EQUITY_EARLY_EXERCISE_UNITS,
    EQUITY_MATHDX_SAMPLE_UNITS,
    EQUITY_SAMPLE_UNITS,
    FIXED_INCOME_UNITS,
    MODEL_SPECS,
    PRODUCT_SPECS,
    SCHEMA_VERSION,
)
from sample_manifest import SAMPLE_MODELS, SampleModelSpec


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


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
    return f'''inline std::vector<ModelParameters> generate_core_parameters(
    std::size_t parameter_count,
    std::uint64_t seed
) {{
    std::vector<ModelParameters> parameters;
    parameters.reserve(parameter_count);
    std::size_t proposal = 0U;
    while (parameters.size() < parameter_count) {{
        HostUniformSequence uniforms(seed, proposal++);
        {uniforms}{derived}
        if (!({model.acceptance})) continue;
        parameters.push_back({model.constructor});
    }}
    return parameters;
}}'''


def _sample_parameter_json(model: SampleModelSpec) -> str:
    entries = ",\n        ".join(
        f'{{"{name}", parameters.{accessor}}}'
        for name, accessor in model.parameters
    )
    return f'''inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {{
    return {{
        {entries}
    }};
}}'''


def _sample_launch_lambda(model: SampleModelSpec) -> str:
    pointers = ",\n            ".join(
        f"outputs[{index}]" for index in range(len(model.outputs))
    )
    prefix = f"model_binding::launch_{model.name}_random_terminal_samples_cuda"
    if model.backend == "volterra":
        geometry = "block_count,"
    else:
        geometry = "threads_per_block, block_count,"
    template = "<factor_count>" if model.backend == "n_factor" else ""
    return f'''[](
        const auto* device_parameters,
        std::size_t parameter_count,
        std::size_t paths_per_parameter,
        std::uint32_t minimum_maturity_days,
        std::uint32_t maximum_maturity_days,
        std::size_t sample_offset,
        std::size_t launch_sample_count,
        unsigned int threads_per_block,
        std::size_t block_count,
        std::uint64_t schedule_seed,
        std::uint64_t dynamics_seed,
        std::uint32_t* device_maturity_days,
        std::span<float*> outputs
    ) {{
        if (outputs.size() != {len(model.outputs)}U) {{
            throw std::invalid_argument("{model.display} sample output arity mismatch.");
        }}
        {prefix}{template}(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, {geometry} schedule_seed, dynamics_seed,
            device_maturity_days,
            {pointers}
        );
    }}'''




def _render_sample_generation_header(model: SampleModelSpec) -> str:
    include_numerics = ""
    generate_call = "generate_model_sample_dataset<ModelParameters>"
    prepare_argument = ""
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
        prepare_argument = '''
        [](const std::vector<ModelParameters>& parameters,
           std::uint32_t maximum_maturity_days) {
            return model_binding::prepare_dynamics<factor_count>(
                parameters,
                static_cast<float>(maximum_maturity_days) / 252.0f,
                1.0f / 504.0f
            );
        },'''
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
    output_metadata = ",\n            ".join(
        f'{{"{name}", {{{{"description", "Terminal {name}."}}, '
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
            "escaped_acceptance": model.acceptance.replace('"', '\\"'),
            "output_metadata": output_metadata,
            "grid": grid,
            "generate_call": generate_call,
            "backend_samples": f"{model.backend}_samples",
            "output_names": ", ".join(
                f'"{name}"' for name in model.outputs
            ),
            "prepare_argument": prepare_argument,
            "launch_lambda": _sample_launch_lambda(model),
        },
    )


def _render_sample_recipe_source(
    model: SampleModelSpec,
    recipe_index: int,
) -> str:
    if recipe_index == 1:
        parameter_count, paths_per_parameter = "12'000U", "250U"
        seed_offsets = (101, 102, 103)
        sample_kind = "conditional"
    else:
        parameter_count, paths_per_parameter = "3'000'000U", "1U"
        seed_offsets = (111, 112, 113)
        sample_kind = "unconditional"
    model_index = SAMPLE_MODELS.index(model) + 1
    seed_prefix = 930_000_000 + model_index * 1_000
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
            "parameter_seed": str(seed_prefix + seed_offsets[0]),
            "schedule_seed": str(seed_prefix + seed_offsets[1]),
            "dynamics_seed": str(seed_prefix + seed_offsets[2]),
        },
    )


def generate_samples(output_root: Path) -> list[Path]:
    generated: list[Path] = []
    for model in SAMPLE_MODELS:
        source_directory = output_root / "src" / "model" / model.source_folder
        source_directory.mkdir(parents=True, exist_ok=True)
        header = source_directory / "sample.cuh"
        source = source_directory / "sample.cu"
        header.write_text(_render_sample_binding(model, "cuh"))
        source.write_text(_render_sample_binding(model, "cu"))
        generated.extend((header, source))

        helper = (
            output_root / "tools" / "sampling" / "generated"
            / f"{model.name}_sample_generation.cuh"
        )
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(_render_sample_generation_header(model))
        generated.append(helper)
        for recipe_index in (1, 2):
            recipe = (
                output_root / "catalog" / "model" / model.asset_class
                / model.name / "samples" / f"samples_{recipe_index:02d}"
                / "generator.cpp"
            )
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(_render_sample_recipe_source(model, recipe_index))
            generated.append(recipe)
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
    for binding in BINDINGS:
        destination = (
            output_root / "src" / "model" / "equity" / "markovian"
            / binding.model / "product"
        )
        destination.mkdir(parents=True, exist_ok=True)
        header = destination / f"{binding.product}.cuh"
        source = destination / f"{binding.product}.cu"
        header.write_text(render(header_template, binding))
        source.write_text(render(source_template, binding))
        generated.extend((header, source))
    analytical_template_dir = (
        TEMPLATE_DIR / "pricing" / "closed_form" / "black_scholes"
    )
    analytical_destination = (
        output_root / "src" / "model" / "equity" / "markovian"
        / "black_scholes" / "product"
    )
    analytical_destination.mkdir(parents=True, exist_ok=True)
    for product in BLACK_SCHOLES_CLOSED_FORM_PRODUCTS:
        for suffix in ("cuh", "cu"):
            destination = analytical_destination / f"{product}.{suffix}"
            template = analytical_template_dir / f"{product}.{suffix}.tpl"
            destination.write_text(template.read_text())
            generated.append(destination)
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
    for binding in ROUGH_PRODUCT_BINDINGS:
        for model, display in ROUGH_VOLTERRA_MODELS:
            values = rough_values(binding, model, display, "volterra")
            destination = (
                output_root / "src" / "model" / "equity" / "rough" / model
                / "product"
            )
            destination.mkdir(parents=True, exist_ok=True)
            header = destination / f"{binding.product}.cuh"
            source = destination / f"{binding.product}.cu"
            header.write_text(templates["volterra_header"].format(**values))
            source.write_text(templates["volterra_source"].format(**values))
            generated.extend((header, source))

        for model, display in ROUGH_N_FACTOR_MODELS:
            values = rough_values(binding, model, display, "n_factor")
            destination = (
                output_root / "src" / "model" / "equity" / "rough" / model
                / "product"
            )
            destination.mkdir(parents=True, exist_ok=True)
            header = destination / f"{binding.product}.cuh"
            source = destination / f"{binding.product}.cu"
            header.write_text(templates["n_factor_header"].format(**values))
            source.write_text(templates["n_factor_source"].format(**values))
            generated.extend((header, source))
    return generated


MONTE_CARLO_PATHS_PER_PRICE = "1'048'576"


def price_recipe_url(model, variant, database_id: str, closed_form: bool) -> str:
    if (
        not closed_form
        and model.legacy_url_name is not None
        and variant.legacy_url_name is not None
    ):
        return (
            "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/"
            f"{model.legacy_url_name}/{variant.legacy_url_name}/"
            f"{database_id}.json"
        )
    return (
        "https://datasets.ai-factory.example/v1/model/equity/"
        f"{model.name}/prices/{variant.name}/{database_id}.json"
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
    for model in MODEL_RECIPE_SPECS:
        for variant in PRICE_VARIANTS:
            database_id = (
                f"{model.name}_01__{variant.name}_01__01"
            )
            closed_form = (
                model.name == "black_scholes"
                and variant.product in BLACK_SCHOLES_CLOSED_FORM_PRODUCTS
            )
            backend = "closed_form" if closed_form else model.backend
            destination = (
                output_root / "catalog" / "model" / "equity" / model.name
                / "prices" / variant.name / database_id / "generator.cpp"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            side_template = (
                f"<OptionSide::{variant.side}>" if variant.side else ""
            )
            product_loader_expression = f"product::{variant.product_loader}"
            if variant.side_aware_loader:
                product_loader_expression = (
                    "[](const auto& path) { return product::"
                    f"{variant.product_loader}(path, OptionSide::{variant.side}); "
                    "}"
                )
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
                    f"datasets/model/equity/{model.name}/parameters/"
                    f"{model.name}_01.json"
                ),
                "product_dataset_path": (
                    "datasets/product/equity/"
                    f"{variant.product_dataset_folder}/"
                    f"{variant.product_dataset_id}.json"
                ),
                "price_dataset_path": (
                    f"datasets/model/equity/{model.name}/prices/"
                    f"{variant.name}/{database_id}.json"
                ),
                "catalog_path": (
                    f"catalog/model/equity/{model.name}/prices/"
                    f"{variant.name}/{database_id}/dataset.yaml"
                ),
                "url": price_recipe_url(
                    model, variant, database_id, closed_form
                ),
                "numerical_method": model.numerical_method,
                "monte_carlo_paths": MONTE_CARLO_PATHS_PER_PRICE,
                "threads_per_block": (
                    "::ai_factory::workbench::offline::cuda_tuning::kMarkovianCompactThreadsPerBlock"
                    if variant.threads_per_block == 256
                    and model.backend == "markovian"
                    else "::ai_factory::workbench::offline::cuda_tuning::kNFactorThreadsPerBlock"
                    if model.backend == "n_factor"
                    else "::ai_factory::workbench::offline::cuda_tuning::kMarkovianThreadsPerBlock"
                ),
                "seed": "910000001" if model.backend == "volterra" else "900000001",
            }
            analytical_steps = variant.analytical_steps_per_day
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
            if backend == "markovian":
                values.update(markovian_time_values(
                    model.name, variant.product
                ))
            destination.write_text(templates[backend].format(**values))
            generated.append(destination)
    for model in AMERICAN_RECIPE_SPECS:
        for side in ("call", "put"):
            database_id = f"{model.model}_01__american_{side}s_01__01"
            destination = (
                output_root / "catalog" / "model" / "equity" / model.model
                / "prices" / f"american_{side}s" / database_id
                / "generator.cpp"
            )
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
            }
            destination.write_text(templates["american"].format(**values))
            generated.append(destination)
    return generated


def cmake_list(name: str, values: list[str]) -> str:
    body = "\n".join(f"    {value}" for value in values)
    return f"set({name}\n{body}\n)\n"


def generate_cmake_manifest(output_root: Path) -> list[Path]:
    """Emit the exact CMake registration matrix from the binding manifest."""
    markovian_models = [model for model, _, _ in MARKOVIAN_MODELS]
    markovian_models.append("black_scholes")
    rough_models = [model for model, _ in ROUGH_MODELS]
    fixed_income_models = [
        model.name for model in MODEL_SPECS
        if model.asset_class == "fixed_income"
    ]
    products = [binding.product for binding in ROUGH_PRODUCT_BINDINGS]
    regular_units = [
        f"{model}/product/{product}"
        for model in markovian_models
        for product in products
    ] + [
        f"{model}/product/{product}"
        for model, _ in ROUGH_N_FACTOR_MODELS
        for product in products
    ]
    volterra_units = [
        f"{model}/product/{product}"
        for model, _ in ROUGH_VOLTERRA_MODELS
        for product in products
    ]
    destination = (
        output_root / "cmake" / "generated" / "EquityPricingBindings.cmake"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Generated by tools/codegen/pricing_bindings/generate.py.\n"
        "# Do not edit: change manifest.py and regenerate the repository.\n\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_EQUITY_MODELS",
            markovian_models + rough_models,
        )
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_FIXED_INCOME_MODELS",
            fixed_income_models,
        )
        + "\n"
        + cmake_list("AI_FACTORY_GENERATED_ROUGH_MODELS", rough_models)
        + "\n"
        + cmake_list(
            "AI_FACTORY_GENERATED_VOLTERRA_MODELS",
            [model for model, _ in ROUGH_VOLTERRA_MODELS],
        )
        + "\n"
        + cmake_list("AI_FACTORY_GENERATED_EQUITY_PRODUCTS", products)
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
    )
    return [destination]


def codegen_source_fingerprint() -> str:
    sources = [
        SCRIPT_DIR / "manifest.py",
        SCRIPT_DIR / "capability_manifest.py",
        SCRIPT_DIR / "sample_manifest.py",
        SCRIPT_DIR / "generate.py",
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
    destination.write_text(json.dumps({
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
    }, indent=2) + "\n")
    return [destination]


def compare(generated: list[Path], output_root: Path, reference_root: Path) -> int:
    mismatch_count = 0
    for generated_path in generated:
        relative_path = generated_path.relative_to(output_root)
        reference_path = reference_root / relative_path
        reference = reference_path.read_text()
        current = generated_path.read_text()
        if current == reference:
            continue
        mismatch_count += 1
        print(f"Mismatch: {relative_path}")
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
            "rough", "markovian", "prototype", "catalog", "samples", "all"
        ),
        default="rough",
    )
    parser.add_argument("--compare-root", type=Path)
    arguments = parser.parse_args()

    generated: list[Path] = []
    if arguments.family in ("markovian", "prototype", "all"):
        generated.extend(generate_markovian(arguments.output))
    if arguments.family in ("rough", "all"):
        generated.extend(generate_rough(arguments.output))
    if arguments.family in ("catalog", "all"):
        generated.extend(generate_catalog_recipes(arguments.output))
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
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
