"""Render gradient wrappers from complete engine-owned C++ templates."""

from pathlib import Path
from string import Template
import json
from .manifest import (
    american_lsm_metadata,
    default_sensitivities,
    terminal_product_metadata,
)
from manifest import PRICE_VARIANTS




FIXED_INCOME_MODEL_DISPLAY = {
    "cir": "CIR",
    "cir_plus_plus": "CIR++",
    "g2": "G2",
    "g2_plus_plus": "G2++",
    "hull_white": "Hull-White",
    "ornstein_uhlenbeck": "OU",
    "vasicek": "Vasicek",
}

CURVE_METADATA = {
    "nelson_siegel": ("Nelson-Siegel", "NelsonSiegel"),
    "svensson": ("Svensson", "Svensson"),
}


def bermudan_lsm_metadata(spec):
    model = spec.pricing.model
    curve = spec.pricing.curve
    terminal = spec.pricing.terminal_forward_dynamics is not None
    fitted = curve is not None
    binding_namespace = model + (f"::{curve}" if fitted else "")
    function_prefix = model + (f"_{curve}" if fitted else "")
    diagnostic_name = model + (f".{curve}" if fitted else "") \
        + ".bermudan_swaption"
    curve_display = CURVE_METADATA[curve][0] if fitted else ""
    curve_type = CURVE_METADATA[curve][1] if fitted else ""
    curve_header_include = (
        f'#include "curve/{curve}/price_gradients/device_preparation.cuh"\n'
        if fitted else ""
    )
    curve_namespace_alias = (
        "namespace curve_pg =\n"
        f"    ::ai_factory::workbench::curve::{curve}::price_gradients;\n"
        if fitted else ""
    )
    curve_type_alias = (
        f"using CurveParameters = curve::{curve}::{curve_type}Parameters;\n"
        if fitted else ""
    )
    plan_type = (
        "fipg::CurveDevicePreparedPlan<\n"
        "    model_pg::DevicePreparation,\n"
        "    curve_pg::DevicePreparation,\n"
        "    bermudan_pg::DevicePreparation\n"
        ">"
        if fitted else
        "fipg::DevicePreparedPlan<\n"
        "    model_pg::DevicePreparation,\n"
        "    bermudan_pg::DevicePreparation\n"
        ">"
    )
    curve_span_argument = (
        f"    std::span<const curve::{curve}::{curve_type}Parameters> curves,\n"
        if fitted else ""
    )
    curve_prepare_argument = "        curves,\n" if fitted else ""
    prepare_function = (
        "fipg::prepare_curve_device_sensitivities"
        if fitted else "fipg::prepare_device_sensitivities"
    )

    if terminal:
        dynamics = "cir::terminal_forward::DynamicsPolicy"
        schedule = "simulation::TerminalForwardRegularExerciseSchedule<Dynamics>"
        continuation = (
            "::ai_factory::workbench::fixed_income::"
            "ScalarRateContinuationState<Dynamics>"
        )
        if fitted:
            central_core = (
                "product::FittedBermudanSwaptionPricingPolicy<\n"
                "        Schedule, CurveParameters, Analytics, Side, "
                "ContinuationState\n"
                "    >"
            )
            bond_analytics = "TerminalForwardBondAnalyticsPolicy"
            implementation_includes = (
                '#include "model/fixed_income/cir/forward_measure_impl.cuh"\n'
                f'#include "model/fixed_income/{model}/{curve}/analytics_impl.cuh"\n'
            )
        else:
            central_core = (
                "product::StandaloneBermudanSwaptionPricingPolicy<\n"
                "        Schedule, Analytics, Side, ContinuationState\n"
                "    >"
            )
            bond_analytics = "terminal_forward::BondAnalyticsPolicy"
            implementation_includes = (
                '#include "model/fixed_income/cir/analytics_impl.cuh"\n'
                '#include "model/fixed_income/cir/forward_measure_impl.cuh"\n'
            )
        central_policy = (
            "product::TerminalForwardBermudanSwaptionPricingPolicy<\n"
            f"    {central_core},\n"
            f"    {bond_analytics}\n"
            ">"
        )
        regressor_basis = (
            "longstaff_schwartz::basis::OneFactorHermiteBasis<3U>"
        )
    else:
        dynamics = {
            "g2": "g2::joint::DynamicsPolicy",
            "g2_plus_plus": "g2_plus_plus::joint::DynamicsPolicy",
            "hull_white": "hull_white::joint::DynamicsPolicy",
            "ornstein_uhlenbeck": "joint::DynamicsPolicy",
            "vasicek": "joint::DynamicsPolicy",
        }[model]
        schedule = "simulation::ExactTransitionRegularExerciseSchedule<Dynamics>"
        two_factor = model in {"g2", "g2_plus_plus"}
        continuation = (
            "::ai_factory::workbench::fixed_income::"
            + ("TwoFactorRateContinuationState<Dynamics>"
               if two_factor else "OneFactorRateContinuationState<Dynamics>")
        )
        central_policy = (
            "product::FittedBermudanSwaptionPricingPolicy<\n"
            "    Schedule, CurveParameters, Analytics, Side, ContinuationState\n"
            ">"
            if fitted else
            "product::StandaloneBermudanSwaptionPricingPolicy<\n"
            "    Schedule, Analytics, Side, ContinuationState\n"
            ">"
        )
        implementation_includes = (
            f'#include "model/fixed_income/{model}/dynamics.cuh"\n'
            f'#include "model/fixed_income/{model}/{curve}/analytics_impl.cuh"\n'
            if fitted else
            f'#include "model/fixed_income/{model}/analytics_impl.cuh"\n'
        )
        regressor_basis = (
            "longstaff_schwartz::basis::TwoFactorHermiteBasis"
            if two_factor else
            "longstaff_schwartz::basis::OneFactorHermiteBasis<3U>"
        )

    return {
        "model_display": FIXED_INCOME_MODEL_DISPLAY[model],
        "curve_display_suffix": f"/{curve_display}" if fitted else "",
        "curve_header_include": curve_header_include,
        "binding_namespace": binding_namespace,
        "curve_namespace_alias": curve_namespace_alias,
        "plan_type": plan_type,
        "function_prefix": function_prefix,
        "curve_span_argument": curve_span_argument,
        "curve_prepare_argument": curve_prepare_argument,
        "prepare_function": prepare_function,
        "unit_path": spec.unit_path.removeprefix("src/"),
        "implementation_includes": implementation_includes,
        "dynamics": dynamics,
        "schedule": schedule,
        "continuation": continuation,
        "curve_type_alias": curve_type_alias,
        "central_policy": central_policy,
        "terminal_forward": "true" if terminal else "false",
        "regressor_basis": regressor_basis,
        "diagnostic_name": diagnostic_name,
    }


def render_bindings(output_root, specifications, template_root, write_generated):
    generated = []
    for spec in specifications:
        product_metadata = (
            terminal_product_metadata(spec.pricing.product)
            if spec.pricing.product in {
                "european_option",
                "asset_or_nothing_option",
                "digital_option",
            }
            else {}
        )
        lsm_metadata = (
            american_lsm_metadata(spec.pricing.model)
            if spec.pricing.product == "american_option"
            else {}
        )
        values = {
            "model": spec.pricing.model,
            "product": spec.pricing.product,
            "product_type": product_metadata.get("product_type", ""),
            "product_policy_header": product_metadata.get(
                "policy_header", ""
            ),
            "sensitivity_policy": product_metadata.get(
                "sensitivity_policy", ""
            ),
            "lsm_schedule": lsm_metadata.get("schedule", ""),
            "lsm_replay_header": lsm_metadata.get("replay_header", ""),
            "lsm_replay": lsm_metadata.get("replay", ""),
            "lsm_continuation": lsm_metadata.get("continuation", ""),
            "lsm_refinement": lsm_metadata.get("refinement", ""),
            "maximum": str(spec.maximum_sensitivities),
            "transition_description": (
                "fixed-step"
                if spec.preparation_strategy
                == "device_prepared_step_terminal"
                else "exact-terminal"
            ),
            "price_only_time_arguments": (
                "            host.time.dt,\n"
                "            host.time.simulation_steps_per_day,\n"
                if spec.preparation_strategy
                == "device_prepared_step_terminal"
                else (
                    "            host.time.dt * static_cast<float>(\n"
                    "                host.time.simulation_steps_per_day\n"
                    "            ),\n"
                )
            ),
        }
        if spec.preparation_strategy == "device_prepared_fixed_income_lsm":
            values.update(bermudan_lsm_metadata(spec))
        for suffix in ("cuh", "cu"):
            if spec.preparation_strategy == "device_prepared_fixed_income_lsm":
                template = (
                    "pricing/longstaff_schwartz/fixed_income/price_gradients/"
                    "bermudan_swaption." + suffix + ".tpl"
                )
            elif spec.preparation_strategy == "device_prepared_lsm":
                template = (
                    "pricing/longstaff_schwartz/equity/"
                    "american_option_price_gradients/product."
                    + suffix + ".tpl"
                )
            elif spec.preparation_strategy == (
                "device_prepared_cooperative_closed_form"
            ):
                template = (
                    "pricing/closed_form/fixed_income/price_gradients/"
                    "european_swaption." + suffix + ".tpl"
                )
            elif spec.preparation_strategy.startswith("device_prepared_"):
                if suffix == "cuh":
                    template = (
                        "pricing/markovian/price_gradients/"
                        "device_prepared_terminal_product.cuh.tpl"
                    )
                else:
                    if spec.preparation_strategy == (
                        "device_prepared_closed_form_terminal"
                    ):
                        template = (
                            "pricing/closed_form/black_scholes/"
                            "price_gradients/"
                            "device_prepared_european_option.cu.tpl"
                        )
                    else:
                        template = (
                            "pricing/markovian/price_gradients/"
                            "device_prepared_monte_carlo_terminal_product."
                            "cu.tpl"
                        )
            elif spec.preparation_strategy == "host_prepared_lsm":
                template = (
                    "pricing/longstaff_schwartz/equity/"
                    "american_option_price_gradients/product."
                    + suffix + ".tpl"
                )
            else:
                raise ValueError(
                    "Unsupported price-gradient preparation strategy: "
                    + spec.preparation_strategy
                )
            destination = output_root / (spec.unit_path + "." + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_generated(destination, Template((template_root / template).read_text()).substitute(values))
            generated.append(destination)
    return generated


def render_recipes(output_root, specifications, sources, model_specs, resolve_rng_domain, rng_mapping_version, template_root, write_generated):
    generated = []
    for spec in specifications:
        if spec.asset_class == "fixed_income":
            template_path = (
                "catalog/pricing/price_gradients/"
                "fixed_income_european_swaption_generator.cpp.tpl"
            )
        else:
            template_path = "catalog/pricing/price_gradients/generator.cpp.tpl"
        if "diagonal_second" in spec.sensitivity_orders:
            template_path = (
                "catalog/pricing/price_gradients/"
                "terminal_diagonal_generator.cpp.tpl"
            )
        if (spec.asset_class == "fixed_income"
                and spec.product == "bermudan_swaption"):
            template_path = (
                "catalog/pricing/price_gradients/"
                "fixed_income_bermudan_swaption_diagonal_generator.cpp.tpl"
                if "diagonal_second" in spec.sensitivity_orders
                else "catalog/pricing/price_gradients/"
                "fixed_income_bermudan_swaption_generator.cpp.tpl"
            )
        elif (spec.asset_class == "fixed_income"
                and "diagonal_second" in spec.sensitivity_orders):
            template_path = (
                "catalog/pricing/price_gradients/"
                "fixed_income_european_swaption_diagonal_generator.cpp.tpl"
            )
        if (spec.asset_class == "equity"
                and spec.product == "american_option"
                and "diagonal_second" in spec.sensitivity_orders):
            template_path = (
                "catalog/pricing/price_gradients/"
                "american_option_diagonal_generator.cpp.tpl"
            )
        elif spec.asset_class == "equity" and spec.product == "american_option":
            template_path = (
                "catalog/pricing/price_gradients/"
                "american_option_generator.cpp.tpl"
            )
        template = Template((template_root / template_path).read_text())
        source = sources[spec.generator_path]
        model = model_specs[spec.model]
        stochastic = spec.engine not in {"equity_closed_form", "fixed_income_closed_form"}
        exact_transition = model.transition_contract == "exact transition"
        selected = default_sensitivities(
            spec.model, spec.product, spec.curve
        )
        seed = resolve_rng_domain(spec).seed("dynamics") if stochastic else 0
        model_input = f"datasets/{model.source_prefix}/parameters/{model.parameter_dataset_id}.json"
        if spec.asset_class == "fixed_income":
            product_input = (
                "datasets/product/bermudan_swaption/"
                "bermudan_swaptions_01.json"
                if spec.product == "bermudan_swaption"
                else "datasets/product/european_swaption/"
                "european_swaptions_01.json"
            )
        elif spec.product == "american_option":
            product_input = (
                "datasets/product/american_option/"
                "american_options_01.json"
            )
        else:
            variant = next(
                item for item in PRICE_VARIANTS
                if item.name == spec.variant
            )
            product_input = (
                f"datasets/product/{spec.product}/"
                f"{variant.product_dataset_id}.json"
            )
        side = ("payer" if "payer" in spec.variant else "receiver") \
            if spec.asset_class == "fixed_income" \
            else "call" if spec.variant.endswith("calls") else "put"
        product_loader_expression = ""
        if spec.asset_class == "equity" and spec.product != "american_option":
            product_loader_expression = (
                f"product::{variant.product_loader}(recipe.product_input"
                + (f", OptionSide::{side}" if variant.side_aware_loader else "")
                + ")"
            )
        fitted = spec.curve is not None
        curve_input = (
            f"datasets/curve/{spec.curve}/{spec.curve}_01.json"
            if fitted else ""
        )
        curve_prefix = f"{spec.curve}/" if fitted else ""
        binding_namespace = spec.model + (
            f"::{spec.curve}" if fitted else ""
        )
        function_prefix = spec.model + (
            f"_{spec.curve}" if fitted else ""
        )
        curve_dataset_include = (
            f'#include "curve/{spec.curve}/dataset.hpp"\n'
            if fitted else ""
        )
        curve_recipe_assignment = (
            f'        recipe.curve_input = "{curve_input}";\n'
            if fitted else ""
        )
        curve_load = (
            f"        const auto curves = curve::{spec.curve}::load_curves("
            "recipe.curve_input);\n"
            if fitted else ""
        )
        curve_lambda_argument = (
            "const auto& curve_rows, " if fitted else ""
        )
        curve_prepare_argument = "curve_rows, " if fitted else ""
        execute_function = (
            "execute_curve_dataset" if fitted else "execute_dataset"
        )
        execute_curve_argument = "curves, " if fitted else ""
        family = (
            "terminal_forward_lsm"
            if spec.product == "bermudan_swaption"
            and spec.model in {"cir", "cir_plus_plus"}
            else "gaussian_rate_lsm"
            if spec.product == "bermudan_swaption"
            else "jamshidian"
            if spec.asset_class == "fixed_income"
            else "closed_form"
            if not stochastic
            else "equity_exact_mc"
            if exact_transition else "equity_step_mc"
        )
        values = {"model": spec.model, "model_input": model_input, "product_input": product_input,
            "product": spec.product,
            "product_loader_expression": product_loader_expression,
            "dataset": spec.dataset_path, "catalog": spec.generation_yaml_path, "url": spec.url,
            "source_recipe": source.recipe_yaml_path, "seed": str(seed), "stochastic": str(stochastic).lower(),
            "side": side,
            "construction": "Aligned" if spec.construction == "aligned" else "CartesianProduct",
            "family": family,
            "exact_transition": str(exact_transition).lower(),
            "curve": spec.curve or "",
            "curve_input": curve_input,
            "curve_prefix": curve_prefix,
            "binding_namespace": binding_namespace,
            "function_prefix": function_prefix,
            "curve_dataset_include": curve_dataset_include,
            "curve_recipe_assignment": curve_recipe_assignment,
            "curve_load": curve_load,
            "curve_lambda_argument": curve_lambda_argument,
            "curve_prepare_argument": curve_prepare_argument,
            "execute_function": execute_function,
            "execute_curve_argument": execute_curve_argument,
            "selections": ",\n        ".join('{"' + item["parameter"] + '", {' + repr(item["displacement"])
                + ', pg::BumpScale::' + item["scale"] + '}}' for item in selected)}
        destination = output_root / spec.generator_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_generated(destination, template.substitute(values))
        generated.append(destination)
        metadata = {"schema_version":1, "kind":"price_gradients", "dataset_id":spec.dataset_id,
            "generator":"generator.cpp", "model_input":model_input, "product_input":product_input,
            "output":{"path":spec.dataset_path,"format":"json"},
            "generation_output":spec.generation_yaml_path, "construction":spec.construction,
            "url":spec.url,
            "paths_per_price":1048576 if stochastic else 0, "dynamics_seed":seed,
            "sensitivity":{"method":"finite_difference_shared_innovations" if stochastic else "finite_difference",
                           "parameters":selected,
                           "source_price_recipe":source.recipe_yaml_path},
            "launch_profile":"gradient candidate; inspect compiled specialization; not performance-qualified"}
        if fitted:
            metadata["curve_input"] = curve_input
        if "diagonal_second" in spec.sensitivity_orders:
            metadata["sensitivity"]["orders"] = list(spec.sensitivity_orders)
        if stochastic:
            metadata["rng_mapping_version"] = rng_mapping_version(spec.model)
        if exact_transition:
            metadata["time_representation"] = {"kind":"exact_terminal_transition",
                "contractual_days_per_year":252,"maturity_bump_steps_per_year":504}
        else:
            metadata["time_grid"] = {"steps_per_year":504,"simulation_steps_per_day":2,"delta_t":"1 / 504"}
        recipe_path = destination.with_name("recipe.yaml")
        write_generated(recipe_path, json.dumps(metadata, indent=2) + "\n")
        generated.append(recipe_path)
    return generated
