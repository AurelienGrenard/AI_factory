#!/usr/bin/env python3
"""Generate thin model-product pricing bindings into an output folder."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from manifest import (
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


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


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
    header_template = (TEMPLATE_DIR / "header.tpl").read_text()
    source_template = (TEMPLATE_DIR / "source.tpl").read_text()
    generated: list[Path] = []
    for binding in BINDINGS:
        destination = (
            output_root / "src" / "model" / "equity" / "markovian"
            / binding.model
        )
        destination.mkdir(parents=True, exist_ok=True)
        header = destination / f"{binding.product}.cuh"
        source = destination / f"{binding.product}.cu"
        header.write_text(render(header_template, binding))
        source.write_text(render(source_template, binding))
        generated.extend((header, source))
    analytical_template_dir = TEMPLATE_DIR / "black_scholes_closed_form"
    analytical_destination = (
        output_root / "src" / "model" / "equity" / "markovian"
        / "black_scholes"
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
            TEMPLATE_DIR / "rough_volterra_header.tpl"
        ).read_text(),
        "volterra_source": (
            TEMPLATE_DIR / "rough_volterra_source.tpl"
        ).read_text(),
        "n_factor_header": (
            TEMPLATE_DIR / "rough_n_factor_header.tpl"
        ).read_text(),
        "n_factor_source": (
            TEMPLATE_DIR / "rough_n_factor_source.tpl"
        ).read_text(),
    }
    generated: list[Path] = []
    for binding in ROUGH_PRODUCT_BINDINGS:
        for model, display in ROUGH_VOLTERRA_MODELS:
            values = rough_values(binding, model, display, "volterra")
            destination = (
                output_root / "src" / "model" / "equity" / "rough" / model
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
            )
            destination.mkdir(parents=True, exist_ok=True)
            header = destination / f"{binding.product}.cuh"
            source = destination / f"{binding.product}.cu"
            header.write_text(templates["n_factor_header"].format(**values))
            source.write_text(templates["n_factor_source"].format(**values))
            generated.extend((header, source))
    return generated


def monte_carlo_paths(model: str, backend: str, variant: str) -> str:
    if backend in ("volterra", "n_factor") or model == "kou":
        return "1'048'576"
    if model == "black_scholes":
        return "65'536"
    if model in ("merton", "variance_gamma") and variant in (
        "european_calls", "european_puts"
    ):
        return "262'144"
    return "16'384"


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
            TEMPLATE_DIR / "catalog_markovian_generator.cpp.tpl"
        ).read_text(),
        "n_factor": (
            TEMPLATE_DIR / "catalog_rough_n_factor_generator.cpp.tpl"
        ).read_text(),
        "volterra": (
            TEMPLATE_DIR / "catalog_rough_volterra_generator.cpp.tpl"
        ).read_text(),
        "closed_form": (
            TEMPLATE_DIR / "catalog_closed_form_generator.cpp.tpl"
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
                "monte_carlo_paths": monte_carlo_paths(
                    model.name, model.backend, variant.name
                ),
                "threads_per_block": str(
                    variant.threads_per_block or model.threads_per_block
                ),
                "seed": "910000001" if model.backend == "volterra" else "900000001",
            }
            analytical_steps = variant.analytical_steps_per_day
            values["analytical_profile_values"] = (
                f"1.0f / {252 * analytical_steps}.0f, 256U, "
                f"{analytical_steps}U"
                if analytical_steps is not None
                else "1.0f / 252.0f, 256U, 0U"
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
    return generated


def cmake_list(name: str, values: list[str]) -> str:
    body = "\n".join(f"    {value}" for value in values)
    return f"set({name}\n{body}\n)\n"


def generate_cmake_manifest(output_root: Path) -> list[Path]:
    """Emit the exact CMake registration matrix from the binding manifest."""
    markovian_models = [model for model, _, _ in MARKOVIAN_MODELS]
    markovian_models.append("black_scholes")
    rough_models = [model for model, _ in ROUGH_MODELS]
    products = [binding.product for binding in ROUGH_PRODUCT_BINDINGS]
    regular_units = [
        f"{model}/{product}"
        for model in markovian_models
        for product in products
    ] + [
        f"{model}/{product}"
        for model, _ in ROUGH_N_FACTOR_MODELS
        for product in products
    ]
    volterra_units = [
        f"{model}/{product}"
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
    )
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
        choices=("rough", "markovian", "prototype", "catalog", "all"),
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
    if arguments.family == "all":
        generated.extend(generate_cmake_manifest(arguments.output))
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
