"""Render gradient wrappers from complete engine-owned C++ templates."""

from pathlib import Path
from string import Template
import json
from .manifest import (
    TERMINAL_PRODUCT_METADATA,
    PATH_PRODUCT_PARAMETER_COUNT,
    BLACK_SCHOLES_CLOSED_FORM_METADATA,
    american_lsm_metadata,
    default_sensitivities,
    terminal_product_metadata,
    path_product_metadata,
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

def terminal_explicit_instantiations(
    model: str,
    product: str,
    product_type: str,
    sided: bool,
    include_mixed: bool,
) -> str:
    sides = ("OptionSide::call", "OptionSide::put") if sided else (None,)
    blocks = []
    for side in sides:
        price_arguments = f"<{side}>" if side else ""
        def sensitivity_arguments(order: str) -> str:
            return (
                f"<{side}, pg::SensitivityOrders::{order}>"
                if side else f"<pg::SensitivityOrders::{order}>"
            )

        lines = []
        if sided:
            lines.extend([
                f"template void launch_{model}_{product}_price_gradients_cuda"
                f"{price_arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::StencilOutputs,",
                "    const pg::LaunchConfiguration&, pg::Outputs);",
            ])
        for order in ("second", "first_and_second"):
            arguments = sensitivity_arguments(order)
            lines.extend([
                f"template void launch_{model}_{product}_diagonal_sensitivities_cuda"
                f"{arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::DiagonalStencilOutputs,",
                "    const pg::LaunchConfiguration&, pg::SensitivityOutputs);",
                "template std::size_t",
                f"{model}_{product}_node_graph_workspace_bytes{arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                "    const pg::LaunchConfiguration&);",
                f"template void launch_{model}_{product}_node_graph_sensitivities_cuda"
                f"{arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::DiagonalStencilOutputs,",
                "    const pg::LaunchConfiguration&, pg::SensitivityOutputs,",
                "    void*, std::size_t);",
            ])
        if sided and include_mixed:
            lines.extend([
                "template std::size_t",
                f"{model}_{product}_mixed_node_graph_workspace_bytes"
                f"<{side}>(",
                f"    const {product_type}PriceGradientPlan&,",
                "    const pg::LaunchConfiguration&);",
                f"template void launch_{model}_{product}_mixed_node_graph_"
                f"sensitivities_cuda<{side}>(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::DiagonalStencilOutputs,",
                f"    {product_type}PriceGradientPlan::MixedStencilOutputs,",
                "    const pg::LaunchConfiguration&,",
                "    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,",
                "    void*, std::size_t);",
            ])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def closed_form_explicit_instantiations(
    model: str, product: str, product_type: str, sided: bool
) -> str:
    sides = ("OptionSide::call", "OptionSide::put") if sided else (None,)
    blocks = []
    for side in sides:
        price_arguments = f"<{side}>" if side else ""
        lines = []
        if sided:
            lines.extend([
                f"template void launch_{model}_{product}_price_gradients_cuda"
                f"{price_arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::StencilOutputs,",
                "    const pg::LaunchConfiguration&, pg::Outputs);",
            ])
        for order in ("second", "first_and_second"):
            arguments = (
                f"<{side}, pg::SensitivityOrders::{order}>"
                if side else f"<pg::SensitivityOrders::{order}>"
            )
            lines.extend([
                f"template void launch_{model}_{product}_diagonal_sensitivities_cuda"
                f"{arguments}(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::DiagonalStencilOutputs,",
                "    const pg::LaunchConfiguration&, pg::SensitivityOutputs);",
            ])
        if side:
            lines.extend([
                "template std::size_t",
                f"{model}_{product}_mixed_node_graph_workspace_bytes<{side}>(",
                f"    const {product_type}PriceGradientPlan&,",
                "    const pg::LaunchConfiguration&);",
                f"template void launch_{model}_{product}_mixed_node_graph_"
                f"sensitivities_cuda<{side}>(",
                f"    const {product_type}PriceGradientPlan&,",
                f"    {product_type}PriceGradientPlan::DeviceInputs,",
                f"    {product_type}PriceGradientPlan::DiagonalStencilOutputs,",
                f"    {product_type}PriceGradientPlan::MixedStencilOutputs,",
                "    const pg::LaunchConfiguration&,",
                "    pg::SensitivityOutputs, pg::MixedSensitivityOutputs,",
                "    void*, std::size_t);",
            ])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


CURVE_METADATA = {
    "flat": ("Flat", "FlatCurve"),
    "nelson_siegel": ("Nelson-Siegel", "NelsonSiegel"),
    "svensson": ("Svensson", "Svensson"),
}


FIXED_INCOME_CLOSED_FORM_PRODUCT_METADATA = {
    "european_swaption": (
        "RegularEuropeanSwaption", "european_swaption"
    ),
    "rate_option": ("RateOption", "rate_option"),
    "zero_coupon_bond_option": (
        "ZeroCouponBondOption", "zero_coupon_bond_option"
    ),
}


def fixed_income_closed_form_metadata(spec):
    model = spec.pricing.model
    curve = spec.pricing.curve
    product = spec.pricing.product
    fitted = curve is not None
    binding_namespace = model + (f"::{curve}" if fitted else "")
    function_prefix = model + (f"_{curve}" if fitted else "")
    diagnostic_name = (
        model + (f".{curve}" if fitted else "") + f".{product}"
    )
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
    curve_span_argument = (
        f"    std::span<const curve::{curve}::{curve_type}Parameters> curves,\n"
        if fitted else ""
    )
    curve_prepare_argument = "        curves,\n" if fitted else ""
    prepare_function = (
        "fipg::prepare_curve_device_sensitivities"
        if fitted else "fipg::prepare_device_sensitivities"
    )
    product_type, product_namespace = (
        FIXED_INCOME_CLOSED_FORM_PRODUCT_METADATA[product]
    )
    plan_type = (
        "fipg::CurveDevicePreparedPlan<\n"
        "    model_pg::DevicePreparation,\n"
        "    curve_pg::DevicePreparation,\n"
        f"    {product_namespace}_pg::DevicePreparation\n"
        ">"
        if fitted else
        "fipg::DevicePreparedPlan<\n"
        "    model_pg::DevicePreparation,\n"
        f"    {product_namespace}_pg::DevicePreparation\n"
        ">"
    )
    if product == "european_swaption":
        pricing_policy = (
            "::ai_factory::workbench::fixed_income::\n"
            "    CooperativeFittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<\n"
            "        Side, FittedAnalyticsProvider, FittedModelComposition,\n"
            "        ModelParameters, CurveParameters,\n"
            "        product::RegularEuropeanSwaptionParameters,\n"
            "        product::RegularEuropeanSwaptionScheduleSource\n"
            "    >"
            if fitted else
            "::ai_factory::workbench::fixed_income::\n"
            "    CooperativeOneFactorEuropeanSwaptionClosedFormPricingPolicy<\n"
            "        Side, AnalyticsProvider, ModelParameters,\n"
            "        product::RegularEuropeanSwaptionParameters,\n"
            "        product::RegularEuropeanSwaptionScheduleSource\n"
            "    >"
        )
        scenario_policy = (
            "::ai_factory::workbench::fixed_income::price_gradients::\n"
            "    CurveScenarioClosedFormPolicy<\n"
            "        PricingPolicy, ModelParameters, CurveParameters,\n"
            "        product::RegularEuropeanSwaptionParameters,\n"
            "        product::RegularEuropeanSwaptionScheduleSource\n"
            "    >"
            if fitted else
            "::ai_factory::workbench::fixed_income::price_gradients::\n"
            "    ScenarioClosedFormPolicy<\n"
            "        PricingPolicy, ModelParameters,\n"
            "        product::RegularEuropeanSwaptionParameters,\n"
            "        product::RegularEuropeanSwaptionScheduleSource\n"
            "    >"
        )
    else:
        policy_stem = (
            "RateOption" if product == "rate_option"
            else "ZeroCouponBondOption"
        )
        pricing_policy = (
            "::ai_factory::workbench::fixed_income::"
            f"Fitted{policy_stem}ClosedFormPricingPolicy<\n"
            "        FittedModelComposition, Side\n"
            "    >"
            if fitted else
            "::ai_factory::workbench::fixed_income::"
            f"Standalone{policy_stem}ClosedFormPricingPolicy<\n"
            "        ModelParameters, Side\n"
            "    >"
        )
        scenario_policy = (
            "::ai_factory::workbench::fixed_income::price_gradients::\n"
            "    CurveScalarScenarioClosedFormPolicy<\n"
            "        PricingPolicy, ModelParameters, CurveParameters,\n"
            f"        product::{product_type}Parameters\n"
            "    >"
            if fitted else
            "::ai_factory::workbench::fixed_income::price_gradients::\n"
            "    ScalarScenarioClosedFormPolicy<\n"
            f"        PricingPolicy, ModelParameters, product::{product_type}Parameters\n"
            "    >"
        )
    implementation_include = (
        f'#include "model/fixed_income/{model}/{curve}/analytics_impl.cuh"'
        if fitted else
        f'#include "model/fixed_income/{model}/analytics_impl.cuh"'
    )
    return {
        "model_display": FIXED_INCOME_MODEL_DISPLAY[model],
        "curve_display_suffix": (
            f"/{CURVE_METADATA[curve][0]}" if fitted else ""
        ),
        "unit_path": spec.unit_path.removeprefix("src/"),
        "binding_namespace": binding_namespace,
        "function_prefix": function_prefix,
        "diagnostic_name": diagnostic_name,
        "curve_header_include": curve_header_include,
        "curve_namespace_alias": curve_namespace_alias,
        "curve_type_alias": curve_type_alias,
        "curve_span_argument": curve_span_argument,
        "curve_prepare_argument": curve_prepare_argument,
        "prepare_function": prepare_function,
        "product_type": product_type,
        "product_namespace": product_namespace,
        "plan_type": plan_type,
        "pricing_policy": pricing_policy,
        "scenario_policy": scenario_policy,
        "implementation_include": implementation_include,
    }


def fixed_income_monte_carlo_metadata(spec):
    values = fixed_income_closed_form_metadata(spec)
    model = spec.pricing.model
    curve = spec.pricing.curve
    fitted = curve is not None
    values.update({
        "sensitivity_policy": (
            "european_swaption_pg::FittedMonteCarloPolicy<\n"
            "        FittedModelComposition, Side\n"
            "    >"
            if fitted else
            "european_swaption_pg::StandaloneMonteCarloPolicy<Side>"
        ),
        "implementation_includes": (
            f'#include "model/fixed_income/{model}/price_gradients/'
            'coupled_dynamics_impl.cuh"\n'
            f'#include "model/fixed_income/{model}/{curve}/analytics_impl.cuh"'
            if fitted else
            f'#include "model/fixed_income/{model}/price_gradients/'
            'coupled_dynamics_impl.cuh"\n'
            f'#include "model/fixed_income/{model}/analytics_impl.cuh"'
        ),
    })
    return values


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
        is_path = spec.preparation_strategy in {
            "device_prepared_path",
            "device_prepared_closed_form_path",
        }
        product_metadata = (
            path_product_metadata(spec.pricing)
            if is_path else
            terminal_product_metadata(spec.pricing.product)
            if spec.pricing.product in TERMINAL_PRODUCT_METADATA
            else {}
        )
        if (spec.pricing.model == "black_scholes"
                and spec.pricing.product
                    in BLACK_SCHOLES_CLOSED_FORM_METADATA):
            product_metadata = {
                **product_metadata,
                **BLACK_SCHOLES_CLOSED_FORM_METADATA[spec.pricing.product],
            }
        lsm_metadata = (
            american_lsm_metadata(spec.pricing.model)
            if spec.pricing.product == "american_option"
            else {}
        )
        values = {
            "model": spec.pricing.model,
            "product": spec.pricing.product,
            "product_type": product_metadata.get("product_type", ""),
            "mixed_output_include": (
                '#include "common/price_gradients/'
                'mixed_sensitivity_outputs.cuh"\n'
                if spec.preparation_strategy in {
                    "device_prepared_step_terminal",
                    "device_prepared_exact_terminal",
                    "device_prepared_path",
                    "device_prepared_closed_form_terminal",
                    "device_prepared_closed_form_path",
                }
                else ""
            ),
            "product_policy_header": product_metadata.get(
                "policy_header", ""
            ),
            "sensitivity_policy": product_metadata.get(
                "sensitivity_policy", ""
            ),
            "closed_form_policy": product_metadata.get(
                "closed_form_policy", ""
            ),
            "schedule": product_metadata.get("schedule", ""),
            "path_exact_transition": (
                "true" if product_metadata.get("exact_transition", False)
                else "false"
            ),
            "calendar_policy": (
                product_metadata.get("sensitivity_policy", "").replace(
                    "<Side>", "<OptionSide::call>"
                )
            ),
            "lsm_schedule": lsm_metadata.get("schedule", ""),
            "lsm_replay_header": lsm_metadata.get("replay_header", ""),
            "lsm_replay": lsm_metadata.get("replay", ""),
            "lsm_continuation": lsm_metadata.get("continuation", ""),
            "lsm_refinement": lsm_metadata.get("refinement", ""),
            "maximum": str(spec.maximum_sensitivities),
            "mixed_maximum": str(
                spec.maximum_sensitivities
                * (spec.maximum_sensitivities - 1) // 2
            ),
            "mixed_nodes_per_worker": str(max(
                2,
                (
                    1 + 3 * spec.maximum_sensitivities
                    + 2 * spec.maximum_sensitivities
                        * (spec.maximum_sensitivities - 1)
                    + 127
                ) // 128,
            )),
            "mixed_team_size": str(next(
                group for group in (2, 4, 8, 16, 32, 64, 128)
                if group * max(
                    2,
                    (
                        1 + 3 * spec.maximum_sensitivities
                        + 2 * spec.maximum_sensitivities
                            * (spec.maximum_sensitivities - 1)
                        + 127
                    ) // 128,
                ) >= (
                    1 + 3 * spec.maximum_sensitivities
                    + 2 * spec.maximum_sensitivities
                        * (spec.maximum_sensitivities - 1)
                )
            )),
            "graph_group_size": str(next(
                group for group in (2, 4, 8, 16, 32)
                if group * 2 >= 1 + 3 * spec.maximum_sensitivities
            )),
            "graph_nodes_per_worker": "2",
            "side_template": (
                "template<OptionSide Side>"
                if product_metadata.get("sided", True) else ""
            ),
            "sensitivity_template": (
                "template<OptionSide Side, pg::SensitivityOrders Orders>"
                if product_metadata.get("sided", True)
                else "template<pg::SensitivityOrders Orders>"
            ),
            "closed_form_helper_template": (
                "template<OptionSide Side, pg::SensitivityOrders Orders, "
                "typename StencilOutputs>"
                if product_metadata.get("sided", True)
                else "template<pg::SensitivityOrders Orders, "
                "typename StencilOutputs>"
            ),
            "closed_form_first_helper_arguments": (
                "Side, pg::SensitivityOrders::first"
                if product_metadata.get("sided", True)
                else "pg::SensitivityOrders::first"
            ),
            "closed_form_diagonal_helper_arguments": (
                "Side, Orders"
                if product_metadata.get("sided", True)
                else "Orders"
            ),
            "side_argument": "<Side>" if product_metadata.get("sided", True) else "",
            "first_variant": (
                'Side == OptionSide::call'
                ' ? "call/nodes=3/B=1" : "put/nodes=3/B=1"'
                if product_metadata.get("sided", True) else '"default/nodes=3/B=1"'
            ),
            "explicit_instantiations": terminal_explicit_instantiations(
                spec.pricing.model,
                spec.pricing.product,
                product_metadata.get("product_type", ""),
                product_metadata.get("sided", True),
                spec.preparation_strategy in {
                    "device_prepared_step_terminal",
                    "device_prepared_exact_terminal",
                    "device_prepared_path",
                    "device_prepared_closed_form_terminal",
                    "device_prepared_closed_form_path",
                },
            ),
            "closed_form_explicit_instantiations": (
                closed_form_explicit_instantiations(
                    spec.pricing.model,
                    spec.pricing.product,
                    product_metadata.get("product_type", ""),
                    product_metadata.get("sided", True),
                )
            ),
            "transition_description": (
                "fixed-step"
                if (
                    spec.pricing.manifest_binding is not None
                    and spec.pricing.manifest_binding.time_kind == "fixed"
                )
                else "exact-terminal"
            ),
            "price_only_time_arguments": (
                "            host.time.dt,\n"
                "            host.time.simulation_steps_per_day,\n"
                if (
                    spec.pricing.manifest_binding is not None
                    and spec.pricing.manifest_binding.time_kind == "fixed"
                )
                else (
                    "            host.time.dt * static_cast<float>(\n"
                    "                host.time.simulation_steps_per_day\n"
                    "            ),\n"
                )
            ),
        }
        node_graph_declarations = ""
        if spec.preparation_strategy in {
            "device_prepared_step_terminal",
            "device_prepared_exact_terminal",
            "device_prepared_path",
        }:
            declaration_template = template_root / (
                "pricing/markovian/price_gradients/"
                + ("device_prepared_path_node_graph_declarations.cuh.tpl"
                   if is_path else
                   "device_prepared_terminal_node_graph_declarations.cuh.tpl")
            )
            node_graph_declarations = Template(
                declaration_template.read_text()
            ).substitute(values)
        elif spec.preparation_strategy in {
            "device_prepared_closed_form_terminal",
            "device_prepared_closed_form_path",
        }:
            declaration_template = template_root / (
                "pricing/closed_form/price_gradients/"
                "mixed_declarations.cuh.tpl"
            )
            node_graph_declarations = Template(
                declaration_template.read_text()
            ).substitute(values)
        values["node_graph_declarations"] = node_graph_declarations
        if spec.preparation_strategy == "device_prepared_fixed_income_lsm":
            values.update(bermudan_lsm_metadata(spec))
        elif spec.preparation_strategy == "device_prepared_fixed_income_monte_carlo":
            values.update(fixed_income_monte_carlo_metadata(spec))
        elif spec.preparation_strategy in {
            "device_prepared_scalar_closed_form",
            "device_prepared_cooperative_closed_form",
        }:
            values.update(fixed_income_closed_form_metadata(spec))
        for suffix in ("cuh", "cu"):
            if spec.preparation_strategy == "device_prepared_fixed_income_lsm":
                template = (
                    "pricing/longstaff_schwartz/fixed_income/price_gradients/"
                    "bermudan_swaption." + suffix + ".tpl"
                )
            elif spec.preparation_strategy == "device_prepared_fixed_income_monte_carlo":
                template = (
                    "pricing/markovian/fixed_income/price_gradients/"
                    "european_swaption." + suffix + ".tpl"
                )
            elif spec.preparation_strategy == "device_prepared_lsm":
                template = (
                    "pricing/longstaff_schwartz/equity/"
                    "american_option_price_gradients/product."
                    + suffix + ".tpl"
                )
            elif spec.preparation_strategy == (
                "device_prepared_scalar_closed_form"
            ):
                template = (
                    "pricing/closed_form/fixed_income/price_gradients/"
                    "scalar_product." + suffix + ".tpl"
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
                        + ("device_prepared_path_product.cuh.tpl"
                           if is_path else
                           "device_prepared_terminal_product.cuh.tpl")
                    )
                else:
                    if spec.preparation_strategy == "device_prepared_path":
                        template = (
                            "pricing/markovian/price_gradients/"
                            "device_prepared_monte_carlo_path_product.cu.tpl"
                        )
                    elif spec.preparation_strategy in {
                        "device_prepared_closed_form_terminal",
                        "device_prepared_closed_form_path",
                    }:
                        template = (
                            "pricing/closed_form/black_scholes/"
                            "price_gradients/"
                            + ("device_prepared_european_option.cu.tpl"
                               if spec.pricing.product == "european_option"
                               else "device_prepared_product.cu.tpl")
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


def early_exercise_numerical_method(spec):
    """Describe the LSM fit shared by both early-exercise replays."""
    if spec.product == "american_option":
        basis = "laguerre_polynomial_two_factor_6_term"
        refinement = american_lsm_metadata(spec.model)["refinement"]
    elif spec.product == "bermudan_swaption":
        basis = (
            "hermite_probabilists_two_factor_quadratic_6_term"
            if spec.model in {"g2", "g2_plus_plus"}
            else "hermite_probabilists_one_factor_degree_3"
        )
        refinement = "none"
    else:
        raise ValueError(
            "Longstaff-Schwartz metadata requested for a non early-exercise "
            f"product: {spec.product}"
        )
    return {
        "engine": spec.engine,
        "algorithm": "longstaff_schwartz",
        "regression": {
            "basis": basis,
            "feature_normalization": "central_lsm_row",
            "solver": "ridge_regularized_fp64_normal_equations_cholesky",
            "refinement": refinement,
        },
    }


def render_recipes(
    output_root,
    specifications,
    sources,
    pricing_bindings,
    model_specs,
    resolve_rng_domain,
    template_root,
    write_generated,
):
    generated = []
    pricing_by_key = {
        (binding.model, binding.curve, binding.product, binding.engine): binding
        for binding in pricing_bindings
    }
    for spec in specifications:
        stochastic = spec.engine not in {
            "equity_closed_form", "fixed_income_closed_form"
        }
        if (spec.asset_class == "fixed_income"
                and spec.product in {
                    "rate_option", "zero_coupon_bond_option"
                }):
            template_path = (
                "catalog/pricing/price_gradients/"
                "fixed_income_scalar_product_generator.cpp.tpl"
            )
        elif spec.asset_class == "fixed_income":
            template_path = (
                "catalog/pricing/price_gradients/"
                "fixed_income_european_swaption_generator.cpp.tpl"
            )
        else:
            template_path = "catalog/pricing/price_gradients/generator.cpp.tpl"
        if "diagonal_second" in spec.sensitivity_orders:
            template_path = (
                "catalog/pricing/price_gradients/"
                + (
                    "monte_carlo_node_graph_diagonal_generator.cpp.tpl"
                    if spec.asset_class == "equity"
                    and spec.product in {
                        *TERMINAL_PRODUCT_METADATA,
                        *PATH_PRODUCT_PARAMETER_COUNT,
                    }
                    and spec.engine != "equity_closed_form"
                    else "terminal_diagonal_generator.cpp.tpl"
                )
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
                + (
                    "fixed_income_scalar_product_diagonal_generator.cpp.tpl"
                    if spec.product in {
                        "rate_option", "zero_coupon_bond_option"
                    }
                    else
                    "fixed_income_european_swaption_node_graph_diagonal_generator.cpp.tpl"
                    if stochastic
                    else "fixed_income_european_swaption_diagonal_generator.cpp.tpl"
                )
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
        if "mixed_second" in spec.sensitivity_orders:
            if spec.asset_class == "equity":
                template_path = (
                    "catalog/pricing/price_gradients/"
                    "american_option_mixed_hessian_generator.cpp.tpl"
                    if spec.product == "american_option"
                    else "catalog/pricing/price_gradients/"
                    "equity_mixed_hessian_generator.cpp.tpl"
                )
            elif spec.product == "bermudan_swaption":
                template_path = (
                    "catalog/pricing/price_gradients/"
                    "fixed_income_bermudan_swaption_mixed_hessian_generator."
                    "cpp.tpl"
                )
            elif spec.product in {
                "rate_option", "zero_coupon_bond_option"
            }:
                template_path = (
                    "catalog/pricing/price_gradients/"
                    "fixed_income_scalar_product_mixed_hessian_generator."
                    "cpp.tpl"
                )
            else:
                template_path = (
                    "catalog/pricing/price_gradients/"
                    "fixed_income_european_swaption_mixed_hessian_generator."
                    "cpp.tpl"
                )
        template = Template((template_root / template_path).read_text())
        source = sources[spec.generator_path]
        model = model_specs[spec.model]
        pricing_binding = pricing_by_key.get(
            (spec.model, spec.curve, spec.product, spec.engine)
        )
        exact_transition = (
            pricing_binding.manifest_binding.time_kind == "exact"
            if pricing_binding is not None
            and pricing_binding.manifest_binding is not None
            else model.transition_contract == "exact transition"
        )
        selected = default_sensitivities(
            spec.model, spec.product, spec.curve
        )
        if spec.sensitivity_parameters:
            by_name = {item["parameter"]: item for item in selected}
            missing = [
                name for name in spec.sensitivity_parameters
                if name not in by_name
            ]
            if missing:
                raise ValueError(
                    f"Unknown sensitivity selection for {spec.generator_path}: "
                    + ", ".join(missing)
                )
            selected = [by_name[name] for name in spec.sensitivity_parameters]
        seed = resolve_rng_domain(spec).seed("dynamics") if stochastic else 0
        model_input = f"datasets/{model.source_prefix}/parameters/{model.parameter_dataset_id}.json"
        if spec.asset_class == "fixed_income":
            fixed_income_product_inputs = {
                "bermudan_swaption": (
                    "datasets/product/bermudan_swaption/"
                    "bermudan_swaptions_01.json"
                ),
                "european_swaption": (
                    "datasets/product/european_swaption/"
                    "european_swaptions_01.json"
                ),
                "rate_option": (
                    "datasets/product/rate_option/rate_options_01.json"
                ),
                "zero_coupon_bond_option": (
                    "datasets/product/zero_coupon_bond_option/"
                    "zero_coupon_bond_options_01.json"
                ),
            }
            product_input = fixed_income_product_inputs[spec.product]
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
        if (spec.asset_class == "fixed_income"
                and spec.product in {
                    "european_swaption", "bermudan_swaption"
                }):
            side = "payer" if "payer" in spec.variant else "receiver"
        elif spec.asset_class == "fixed_income":
            side = {
                "caplets": "call",
                "floorlets": "put",
                "zero_coupon_bond_calls": "call",
                "zero_coupon_bond_puts": "put",
            }[spec.variant]
        else:
            side = "call" if spec.variant.endswith("calls") else "put"
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
            else "fixed_income_mc"
            if spec.asset_class == "fixed_income" and stochastic
            else "jamshidian"
            if spec.asset_class == "fixed_income"
            and spec.product == "european_swaption"
            else "closed_form"
            if spec.asset_class == "fixed_income"
            else "closed_form"
            if not stochastic
            else "equity_exact_mc"
            if exact_transition else "equity_step_mc"
        )
        values = {"model": spec.model, "model_input": model_input, "product_input": product_input,
            "product": spec.product,
            "product_loader": {
                "rate_option": "load_rate_options",
                "zero_coupon_bond_option": "load_zero_coupon_bond_options",
            }.get(spec.product, ""),
            "product_loader_expression": product_loader_expression,
            "dataset": spec.dataset_path, "catalog": spec.generation_yaml_path, "url": spec.url,
            "source_recipe": source.recipe_yaml_path, "seed": str(seed), "stochastic": str(stochastic).lower(),
            "side": side,
            "price_template_arguments": (
                f"<OptionSide::{side}>" if variant.side is not None else ""
            ) if spec.asset_class == "equity"
                and spec.product != "american_option" else "",
            "sensitivity_template_arguments": (
                f"<OptionSide::{side}, "
                "pg::SensitivityOrders::first_and_second>"
                if variant.side is not None
                else "<pg::SensitivityOrders::first_and_second>"
            ) if spec.asset_class == "equity"
                and spec.product != "american_option" else "",
            "construction": "Aligned" if spec.construction == "aligned" else "CartesianProduct",
            "family": family,
            "exact_transition": str(exact_transition).lower(),
            "exercise_replay": spec.exercise_replay or "",
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
            "execute_curve_prefix": "execute_curve_" if fitted else "execute_",
            "closed_form_distribution_argument": (
                ",\n                    closed_form::WorkDistribution::cooperative"
                if not stochastic else ""
            ),
            "mixed_distribution_argument": (
                ",\n                        "
                "closed_form::WorkDistribution::cooperative"
                if not stochastic else ""
            ),
            "paths_per_price": (
                "offline::cuda_tuning::kProductionPathsPerPrice"
                if stochastic else "0U"
            ),
            "selections": ",\n        ".join('{"' + item["parameter"] + '", {' + repr(item["displacement"])
                + ', pg::BumpScale::' + item["scale"] + '}}' for item in selected)}
        from tools.datasets.catalog_layout import physical_path
        project_root = Path(__file__).resolve().parents[4]
        relative = physical_path(project_root, spec.generator_path).relative_to(project_root)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_generated(destination, template.substitute(values))
        generated.append(destination)
        inputs = {"model": model_input, "product": product_input}
        if fitted:
            inputs["curve"] = curve_input
        metadata = {
            "schema_version": 1,
            "kind": "price_gradients",
            "dataset_id": spec.dataset_id,
            "generator": "generator.cpp",
            "inputs": inputs,
            "output": {"path": spec.dataset_path, "format": "json"},
            "generation_output": spec.generation_yaml_path,
            "url": spec.url,
            "row_count": spec.row_count,
            "paths_per_price": 1048576 if stochastic else 0,
            "sensitivity": {
                "method": (
                    "finite_difference_shared_innovations"
                    if stochastic else "finite_difference"
                ),
                "parameters": selected,
                "source_price_recipe": source.recipe_yaml_path,
            },
        }
        if spec.exercise_replay is not None:
            metadata["numerical_method"] = early_exercise_numerical_method(spec)
            metadata["exercise_replay"] = spec.exercise_replay
        if stochastic:
            metadata["seeds"] = {"dynamics": seed}
            metadata["random_number_generator"] = "philox"
        if "diagonal_second" in spec.sensitivity_orders:
            metadata["sensitivity"]["orders"] = list(spec.sensitivity_orders)
        if "mixed_second" in spec.sensitivity_orders:
            metadata["sensitivity"]["mixed_second"] = "all"
        if exact_transition:
            metadata["time_representation"] = {
                "kind": (
                    "exact_schedule_transition"
                    if spec.product in PATH_PRODUCT_PARAMETER_COUNT
                    else "exact_terminal_transition"
                ),
                "contractual_days_per_year": 252,
                "maturity_bump_steps_per_year": 504,
            }
        else:
            metadata["time_grid"] = {"steps_per_year":504,"simulation_steps_per_day":2,"delta_t":"1 / 504"}
        recipe_path = destination.with_name("recipe.yaml")
        write_generated(recipe_path, json.dumps(metadata, indent=2) + "\n")
        generated.append(recipe_path)
    return generated
