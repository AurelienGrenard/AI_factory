"""Derive the implemented gradient integration surface from existing pricing bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capability_manifest import ProductBindingSpec


TERMINAL_PRODUCT_METADATA = {
    "european_option": {
        "product_type": "EuropeanOption",
        "parameter_count": 1,
        "policy_header": (
            "product/european_option/price_gradients/monte_carlo_policy.cuh"
        ),
        "sensitivity_policy": (
            "product::EuropeanOptionGradientPathPolicy<Side>"
        ),
    },
    "asset_or_nothing_option": {
        "product_type": "AssetOrNothingOption",
        "parameter_count": 1,
        "policy_header": "product/asset_or_nothing_option/pricing_policy.cuh",
        "sensitivity_policy": (
            "epg::TerminalProductSensitivityPolicy<"
            "product::AssetOrNothingOptionPathPolicy<Side>>"
        ),
    },
    "digital_option": {
        "product_type": "DigitalOption",
        "parameter_count": 2,
        "policy_header": "product/digital_option/pricing_policy.cuh",
        "sensitivity_policy": (
            "epg::TerminalProductSensitivityPolicy<"
            "product::DigitalOptionPathPolicy<Side>>"
        ),
    },
    "gap_option": {
        "product_type": "GapOption",
        "parameter_count": 2,
        "policy_header": "product/gap_option/pricing_policy.cuh",
        "sensitivity_policy": (
            "epg::TerminalProductSensitivityPolicy<"
            "product::GapOptionPathPolicy<Side>>"
        ),
    },
    "straddle": {
        "product_type": "Straddle",
        "parameter_count": 1,
        "sided": False,
        "policy_header": "product/straddle/pricing_policy.cuh",
        "sensitivity_policy": (
            "epg::TerminalProductSensitivityPolicy<"
            "product::StraddlePathPolicy>"
        ),
    },
}


BLACK_SCHOLES_CLOSED_FORM_METADATA = {
    "asset_or_nothing_option": {
        "product_type": "AssetOrNothingOption",
        "closed_form_policy":
            "AssetOrNothingOptionClosedFormPricingPolicy<Side>",
        "sided": True,
    },
    "digital_option": {
        "product_type": "DigitalOption",
        "closed_form_policy": "DigitalOptionClosedFormPricingPolicy<Side>",
        "sided": True,
    },
    "gap_option": {
        "product_type": "GapOption",
        "closed_form_policy": "GapOptionClosedFormPricingPolicy<Side>",
        "sided": True,
    },
    "straddle": {
        "product_type": "Straddle",
        "closed_form_policy": "StraddleClosedFormPricingPolicy",
        "sided": False,
    },
    "forward_start_option": {
        "product_type": "ForwardStartOption",
        "closed_form_policy":
            "ForwardStartOptionClosedFormPricingPolicy<Side>",
        "sided": True,
        "exact_transition": True,
    },
    "geometric_asian_option": {
        "product_type": "GeometricAsianOption",
        "closed_form_policy":
            "GeometricAsianOptionClosedFormPricingPolicy<Side>",
        "sided": True,
        "exact_transition": False,
    },
    "range_accrual": {
        "product_type": "RangeAccrual",
        "closed_form_policy": "RangeAccrualClosedFormPricingPolicy",
        "sided": False,
        "exact_transition": True,
    },
}


PATH_PRODUCT_PARAMETER_COUNT = {
    "asian_option": 1,
    "athena_autocall": 3,
    "cliquet": 5,
    "double_knock_out_option": 3,
    "down_and_in_option": 2,
    "down_and_out_option": 2,
    "forward_start_option": 1,
    "geometric_asian_option": 1,
    "lookback_option": 1,
    "phoenix_autocall": 4,
    "phoenix_memory_autocall": 4,
    "range_accrual": 3,
    "up_and_in_option": 2,
    "up_and_out_option": 2,
    "up_no_touch": 2,
    "up_one_touch": 2,
}


def path_product_metadata(binding):
    manifest = binding.manifest_binding
    if manifest is None:
        if (binding.model == "black_scholes"
                and binding.product in BLACK_SCHOLES_CLOSED_FORM_METADATA):
            metadata = BLACK_SCHOLES_CLOSED_FORM_METADATA[binding.product]
            return {
                **metadata,
                "parameter_count": PATH_PRODUCT_PARAMETER_COUNT[
                    binding.product
                ],
                "policy_header": "",
                "sensitivity_policy": "",
                "schedule": "",
            }
        raise KeyError(binding.product)
    if binding.product not in PATH_PRODUCT_PARAMETER_COUNT:
        raise KeyError(binding.product)
    path_policy = f"product::{manifest.pricing_policy.replace('PricingPolicy', 'PathPolicy')}"
    if manifest.sided:
        path_policy += "<Side>"
    schedule = f"{manifest.schedule}<{binding.model}::DynamicsPolicy"
    if manifest.schedule.endswith("CalendarSchedule"):
        schedule += ", 2U"
    schedule += ">"
    return {
        "product_type": manifest.product_type,
        "parameter_count": PATH_PRODUCT_PARAMETER_COUNT[binding.product],
        "policy_header": f"product/{binding.product}/pricing_policy.cuh",
        "sensitivity_policy": path_policy,
        "schedule": schedule,
        "exact_transition": "ExactTransition" in manifest.schedule,
        "sided": manifest.sided,
    }


AMERICAN_LSM_METADATA = {
    "bates": {
        "schedule": "FixedStepMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_fixed_step_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedFixedStepFrozenExerciseReplay",
        "continuation": (
            "product::SpotAndScaledStateContinuationState<"
            "bates::DynamicsPolicy, &bates::State::variance, "
            "&bates::ModelParameters::theta>"
        ),
        "refinement": "none",
    },
    "black_scholes": {
        "schedule": "ExactTransitionMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_exact_transition_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedExactTransitionFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "black_scholes::DynamicsPolicy>"
        ),
        "refinement": "none",
    },
    "cev": {
        "schedule": "FixedStepMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_fixed_step_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedFixedStepFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "cev::DynamicsPolicy>"
        ),
        "refinement": "none",
    },
    "heston": {
        "schedule": "FixedStepMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_fixed_step_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedFixedStepFrozenExerciseReplay",
        "continuation": (
            "product::SpotAndScaledStateContinuationState<"
            "heston::DynamicsPolicy, &heston::State::variance, "
            "&heston::ModelParameters::theta>"
        ),
        "refinement": "none",
    },
    "kou": {
        "schedule": "ExactTransitionMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_exact_transition_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedExactTransitionFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "kou::DynamicsPolicy>"
        ),
        "refinement": "normal_residual",
    },
    "merton": {
        "schedule": "ExactTransitionMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_exact_transition_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedExactTransitionFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "merton::DynamicsPolicy>"
        ),
        "refinement": "none",
    },
    "normal_inverse_gaussian": {
        "schedule": "ExactTransitionMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_exact_transition_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedExactTransitionFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "normal_inverse_gaussian::DynamicsPolicy>"
        ),
        "refinement": "none",
    },
    "schobel_zhu": {
        "schedule": "FixedStepMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_fixed_step_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedFixedStepFrozenExerciseReplay",
        "continuation": (
            "product::SpotAndScaledStateContinuationState<"
            "schobel_zhu::DynamicsPolicy, "
            "&schobel_zhu::State::volatility, "
            "&schobel_zhu::ModelParameters::long_run_volatility>"
        ),
        "refinement": "none",
    },
    "variance_gamma": {
        "schedule": "ExactTransitionMaturityAlignedExerciseSchedule",
        "replay_header": (
            "device_prepared_exact_transition_frozen_exercise_replay.cuh"
        ),
        "replay": "DevicePreparedExactTransitionFrozenExerciseReplay",
        "continuation": (
            "product::SpotLogMoneynessContinuationState<"
            "variance_gamma::DynamicsPolicy>"
        ),
        "refinement": "none",
    },
}


def terminal_product_metadata(product: str):
    return TERMINAL_PRODUCT_METADATA[product]


def american_lsm_metadata(model: str):
    return AMERICAN_LSM_METADATA[model]


@dataclass(frozen=True)
class PriceGradientBindingSpec:
    pricing: ProductBindingSpec
    maximum_sensitivities: int
    closed_form: bool
    preparation_strategy: str
    supported_orders: tuple[str, ...] = ("first",)

    def __post_init__(self):
        supported = {
            "device_prepared_closed_form_terminal",
            "device_prepared_closed_form_path",
            "device_prepared_step_terminal",
            "device_prepared_exact_terminal",
            "device_prepared_path",
            "device_prepared_lsm",
            "device_prepared_fixed_income_lsm",
            "host_prepared_lsm",
            "device_prepared_cooperative_closed_form",
            "device_prepared_scalar_closed_form",
            "device_prepared_fixed_income_monte_carlo",
        }
        if self.preparation_strategy not in supported:
            raise ValueError(
                "Unknown price-gradient preparation strategy: "
                + self.preparation_strategy
            )
        if self.supported_orders not in {
            ("first",),
            ("first", "diagonal_second", "mixed_second"),
        }:
            raise ValueError("Invalid price-gradient derivative orders")

    @property
    def unit_path(self):
        return self.pricing.unit_path + "_price_gradients"

    @property
    def paths(self):
        return tuple(self.unit_path + "." + suffix for suffix in ("cuh", "cu"))


def compose_bindings(pricing_bindings):
    """Expose only implemented engines; further models are added after their coupling."""
    model_parameter_count = {
        "bates": 11,
        "black_scholes": 4,
        "heston": 8,
        "heston_3_2": 8,
        "cev": 5,
        "kou": 8,
        "merton": 7,
        "normal_inverse_gaussian": 6,
        "sabr": 7,
        "schobel_zhu": 8,
        "stein_stein": 7,
        "variance_gamma": 6,
    }
    supports_maturity = {model: True for model in model_parameter_count}
    terminal_strategy = {
        "bates": "device_prepared_step_terminal",
        "black_scholes": "device_prepared_closed_form_terminal",
        "cev": "device_prepared_step_terminal",
        "heston": "device_prepared_step_terminal",
        "heston_3_2": "device_prepared_step_terminal",
        "kou": "device_prepared_exact_terminal",
        "merton": "device_prepared_exact_terminal",
        "normal_inverse_gaussian": "device_prepared_exact_terminal",
        "sabr": "device_prepared_step_terminal",
        "schobel_zhu": "device_prepared_step_terminal",
        "stein_stein": "device_prepared_step_terminal",
        "variance_gamma": "device_prepared_exact_terminal",
    }
    lsm_models = set(AMERICAN_LSM_METADATA)
    terminal_equity = tuple(
        PriceGradientBindingSpec(
            binding,
            model_parameter_count[binding.model]
                + TERMINAL_PRODUCT_METADATA[binding.product]["parameter_count"]
                + supports_maturity[binding.model],
            binding.model == "black_scholes",
            terminal_strategy[binding.model],
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.model in model_parameter_count
        and binding.product in TERMINAL_PRODUCT_METADATA
    )
    path_equity = tuple(
        PriceGradientBindingSpec(
            binding,
            model_parameter_count[binding.model]
                + PATH_PRODUCT_PARAMETER_COUNT[binding.product]
                + 1,
            False,
            "device_prepared_path",
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.model in model_parameter_count
        and binding.product in PATH_PRODUCT_PARAMETER_COUNT
        and binding.engine == "equity_markovian"
    )
    closed_form_path_equity = tuple(
        PriceGradientBindingSpec(
            binding,
            model_parameter_count[binding.model]
                + PATH_PRODUCT_PARAMETER_COUNT[binding.product]
                + 1,
            True,
            "device_prepared_closed_form_path",
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.model == "black_scholes"
        and binding.product in PATH_PRODUCT_PARAMETER_COUNT
        and binding.engine == "equity_closed_form"
    )
    lsm_equity = tuple(
        PriceGradientBindingSpec(
            binding,
            model_parameter_count[binding.model] + 1,
            False,
            "device_prepared_lsm",
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.model in lsm_models
        and binding.product == "american_option"
    )
    equity = (
        *terminal_equity,
        *path_equity,
        *closed_form_path_equity,
        *lsm_equity,
    )
    fixed_income_model_parameter_count = {
        "cir": 4,
        "cir_plus_plus": 4,
        "g2": 7,
        "g2_plus_plus": 5,
        "hull_white": 2,
        "ornstein_uhlenbeck": 3,
        "vasicek": 4,
    }
    curve_parameter_count = {
        None: 0,
        "flat": 1,
        "nelson_siegel": 4,
        "svensson": 6,
    }
    fixed_income_product_parameter_count = {
        "european_swaption": 3,
        "rate_option": 2,
        "zero_coupon_bond_option": 2,
    }
    fixed_income_closed_form = tuple(
        PriceGradientBindingSpec(
            binding,
            fixed_income_model_parameter_count[binding.model]
                + curve_parameter_count[binding.curve]
                + fixed_income_product_parameter_count[binding.product]
                + 1,
            True,
            ("device_prepared_cooperative_closed_form"
             if binding.product == "european_swaption"
             else "device_prepared_scalar_closed_form"),
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.asset_class == "fixed_income"
        and binding.engine == "fixed_income_closed_form"
        and binding.product in fixed_income_product_parameter_count
    )
    fixed_income_monte_carlo = tuple(
        PriceGradientBindingSpec(
            binding,
            fixed_income_model_parameter_count[binding.model]
                + curve_parameter_count[binding.curve]
                + fixed_income_product_parameter_count[binding.product]
                + 1,
            False,
            "device_prepared_fixed_income_monte_carlo",
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.asset_class == "fixed_income"
        and binding.engine == "fixed_income_monte_carlo"
        and binding.product == "european_swaption"
    )
    fixed_income_lsm = tuple(
        PriceGradientBindingSpec(
            binding,
            fixed_income_model_parameter_count[binding.model]
            + curve_parameter_count[binding.curve]
            + 3,
            False,
            "device_prepared_fixed_income_lsm",
            ("first", "diagonal_second", "mixed_second"),
        )
        for binding in pricing_bindings
        if binding.asset_class == "fixed_income"
        and binding.product == "bermudan_swaption"
        and binding.engine == "fixed_income_lsm"
    )
    return (
        *equity,
        *fixed_income_closed_form,
        *fixed_income_monte_carlo,
        *fixed_income_lsm,
    )


def compose_datasets(price_datasets, bindings):
    """Compose every supported price-gradient dataset variant.

    The public catalogue selects a deliberately small subset of these specs.
    Keeping the other variants here preserves code-generation coverage without
    publishing four near-identical recipes for every pricing binding.
    """
    supported = {(spec.pricing.model, spec.pricing.product) for spec in bindings}
    diagonal_pairs = {
        (spec.pricing.model, spec.pricing.product) for spec in bindings
        if "diagonal_second" in spec.supported_orders
        and spec.preparation_strategy.startswith("device_prepared_")
    }
    equity = tuple(replace(dataset,
        dataset_id=dataset.dataset_id + "_price_gradient",
        dataset_kind="price_gradients",
        generator_path=dataset.generator_path.replace(
            "/prices/", "/price_gradients/"
        ).replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id + "_price_gradient/",
        ),
        template="catalog/pricing/price_gradients/generator.cpp.tpl",
        numerical_profile="selected_gradients_production_paths",
        layout="row_major_selected_gradients",
        exercise_replay=(
            "frozen_exercise_time"
            if dataset.product == "american_option" else None
        ),
        sensitivity_orders=("first",))
        for dataset in price_datasets
        if dataset.dataset_kind == "prices"
        and dataset.asset_class == "equity"
        and (dataset.model, dataset.product) in supported)
    spot = tuple(replace(
        dataset,
        dataset_id=(
            dataset.dataset_id.removesuffix("_price_gradient")
            + "_price_gradient_spot"
        ),
        generator_path=dataset.generator_path.replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id.removesuffix("_price_gradient")
            + "_price_gradient_spot/",
        ),
        sensitivity_parameters=("model.spot",),
    ) for dataset in equity)
    fixed_income = tuple(replace(dataset,
        dataset_id=dataset.dataset_id + "_price_gradient",
        dataset_kind="price_gradients",
        generator_path=dataset.generator_path.replace(
            "/prices/", "/price_gradients/"
        ).replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id + "_price_gradient/",
        ),
        template=(
            "catalog/pricing/price_gradients/"
            "fixed_income_bermudan_swaption_generator.cpp.tpl"
            if dataset.product == "bermudan_swaption"
            else "catalog/pricing/price_gradients/"
            "fixed_income_scalar_product_generator.cpp.tpl"
            if dataset.product in {
                "rate_option", "zero_coupon_bond_option"
            }
            else "catalog/pricing/price_gradients/"
            "fixed_income_european_swaption_generator.cpp.tpl"
        ),
        numerical_profile=(
            "selected_gradients_production_paths"
            if dataset.product == "bermudan_swaption"
            else "selected_gradients_closed_form"
        ),
        layout="row_major_selected_gradients",
        exercise_replay=(
            "frozen_exercise_time"
            if dataset.product == "bermudan_swaption" else None
        ),
        sensitivity_orders=("first",))
        for dataset in price_datasets
        if dataset.dataset_kind == "prices"
        and dataset.asset_class == "fixed_income"
        and dataset.product in {
            "european_swaption", "bermudan_swaption",
            "rate_option", "zero_coupon_bond_option",
        }
        and (dataset.model, dataset.product) in supported
        and any(spec.pricing.model == dataset.model
                and spec.pricing.product == dataset.product
                and spec.pricing.curve == dataset.curve
                for spec in bindings))
    diagonal = tuple(replace(dataset,
        dataset_id=dataset.dataset_id + "_diagonal_hessian",
        generator_path=dataset.generator_path.replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id + "_diagonal_hessian/"),
        template=("catalog/pricing/price_gradients/fixed_income_bermudan_swaption_diagonal_generator.cpp.tpl"
                  if dataset.product == "bermudan_swaption"
                  else "catalog/pricing/price_gradients/fixed_income_scalar_product_diagonal_generator.cpp.tpl"
                  if dataset.asset_class == "fixed_income"
                  and dataset.product in {
                      "rate_option", "zero_coupon_bond_option"
                  }
                  else "catalog/pricing/price_gradients/fixed_income_european_swaption_diagonal_generator.cpp.tpl"
                  if dataset.asset_class == "fixed_income"
                  else "catalog/pricing/price_gradients/american_option_diagonal_generator.cpp.tpl"
                  if dataset.product == "american_option"
                  else "catalog/pricing/price_gradients/terminal_diagonal_generator.cpp.tpl"),
        numerical_profile="selected_first_and_diagonal_second_production_paths",
        sensitivity_orders=("first", "diagonal_second"))
        for dataset in (*equity, *fixed_income)
        if (dataset.model, dataset.product) in diagonal_pairs)
    frozen_policy_diagonal = tuple(
        replace(
            dataset,
            dataset_id=dataset.dataset_id + "_frozen_policy",
            generator_path=dataset.generator_path.replace(
                "/" + dataset.dataset_id + "/",
                "/" + dataset.dataset_id + "_frozen_policy/",
            ),
            exercise_replay="frozen_regression_policy",
        )
        for dataset in diagonal
        if dataset.product in {"american_option", "bermudan_swaption"}
    )
    full_hessian = tuple(replace(
        dataset,
        dataset_id=dataset.dataset_id.removesuffix("_diagonal_hessian")
            + "_full_hessian",
        generator_path=dataset.generator_path.replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id.removesuffix("_diagonal_hessian")
                + "_full_hessian/",
        ),
        numerical_profile="selected_full_hessian_production_paths",
        layout="row_major_selected_full_hessian",
        sensitivity_orders=(
            "first", "diagonal_second", "mixed_second"
        ),
    ) for dataset in diagonal)
    return (
        *spot,
        *equity,
        *fixed_income,
        *diagonal,
        *frozen_policy_diagonal,
        *full_hessian,
    )


def default_sensitivities(
    model,
    product="european_option",
    curve=None,
):
    """Initial recipes select useful coordinates without inferring inclusion from zero values."""
    coordinates = {
        "bates": (("model.spot", .005, "relative"),
                  ("model.initial_variance", .001, "absolute"),
                  ("model.rho", .002, "absolute"),
                  ("model.jump_intensity", .05, "absolute"),
                  ("model.jump_log_mean", .002, "absolute")),
        "black_scholes": (("model.spot", .005, "relative"),
                          ("model.volatility", .005, "relative")),
        "heston": (("model.spot", .005, "relative"),
                   ("model.initial_variance", .001, "absolute"),
                   ("model.rho", .002, "absolute")),
        "heston_3_2": (("model.spot", .005, "relative"),
                       ("model.initial_variance", .001, "absolute"),
                       ("model.rho", .002, "absolute")),
        "cev": (("model.spot", .005, "relative"),
                ("model.sigma", .005, "relative"),
                ("model.beta", .002, "absolute")),
        "merton": (("model.spot", .005, "relative"), ("model.volatility", .005, "relative"),
                   ("model.jump_log_mean", .002, "absolute")),
        "normal_inverse_gaussian": (
            ("model.spot", .005, "relative"),
            ("model.alpha", .005, "relative"),
            ("model.beta", .002, "absolute"),
            ("model.delta", .005, "relative"),
        ),
        "kou": (("model.spot", .005, "relative"),
                ("model.volatility", .005, "relative"),
                ("model.jump_intensity", .05, "absolute"),
                ("model.up_probability", .002, "absolute")),
        "sabr": (("model.spot", .005, "relative"),
                 ("model.initial_volatility", .005, "relative"),
                 ("model.rho", .002, "absolute"),
                 ("model.beta", .002, "absolute")),
        "schobel_zhu": (("model.spot", .005, "relative"),
                        ("model.initial_volatility", .005, "relative"),
                        ("model.correlation", .002, "absolute")),
        "stein_stein": (("model.spot", .005, "relative"),
                        ("model.initial_volatility", .005, "relative"),
                        ("model.rho", .002, "absolute")),
        "variance_gamma": (("model.spot", .005, "relative"),
                           ("model.sigma", .005, "relative"),
                           ("model.nu", .005, "relative"),
                           ("model.theta", .002, "absolute")),
        "cir": (("model.mean_reversion", .005, "relative"),
                ("model.long_term_mean", .005, "relative"),
                ("model.volatility", .005, "relative"),
                ("model.initial_state", .0005, "absolute")),
        "cir_plus_plus": (("model.mean_reversion", .005, "relative"),
                           ("model.long_term_mean", .005, "relative"),
                           ("model.volatility", .005, "relative"),
                           ("model.initial_state", .0005, "absolute")),
        "g2": (("model.mean_reversion_x", .005, "relative"),
               ("model.volatility_x", .005, "relative"),
               ("model.mean_reversion_y", .005, "relative"),
               ("model.volatility_y", .005, "relative"),
               ("model.correlation", .002, "absolute"),
               ("model.initial_state_x", .0005, "absolute"),
               ("model.initial_state_y", .0005, "absolute")),
        "g2_plus_plus": (("model.mean_reversion_x", .005, "relative"),
                         ("model.volatility_x", .005, "relative"),
                         ("model.mean_reversion_y", .005, "relative"),
                         ("model.volatility_y", .005, "relative"),
                         ("model.correlation", .002, "absolute")),
        "hull_white": (("model.mean_reversion", .005, "relative"),
                       ("model.volatility", .005, "relative")),
        "ornstein_uhlenbeck": (
            ("model.mean_reversion", .005, "relative"),
            ("model.volatility", .005, "relative"),
            ("model.initial_state", .0005, "absolute"),
        ),
        "vasicek": (("model.mean_reversion", .005, "relative"),
                    ("model.long_term_mean", .0005, "absolute"),
                    ("model.volatility", .005, "relative"),
                    ("model.initial_state", .0005, "absolute")),
    }[model]
    curves = {
        None: (),
        "flat": (
            ("curve.rate", .0005, "absolute"),
        ),
        "nelson_siegel": (
            ("curve.beta0", .0005, "absolute"),
            ("curve.beta1", .0005, "absolute"),
            ("curve.beta2", .0005, "absolute"),
            ("curve.tau", .005, "relative"),
        ),
        "svensson": (
            ("curve.beta0", .0005, "absolute"),
            ("curve.beta1", .0005, "absolute"),
            ("curve.beta2", .0005, "absolute"),
            ("curve.beta3", .0005, "absolute"),
            ("curve.tau1", .005, "relative"),
            ("curve.tau2", .005, "relative"),
        ),
    }[curve]
    products = {
        "european_option": (("product.strike", .005, "relative"),),
        "american_option": (("product.strike", .005, "relative"),),
        "asset_or_nothing_option": (
            ("product.strike", .005, "relative"),
        ),
        "digital_option": (
            ("product.strike", .005, "relative"),
            ("product.cash_payoff", .005, "relative"),
        ),
        "gap_option": (
            ("product.trigger_strike", .005, "relative"),
            ("product.payoff_strike", .005, "relative"),
        ),
        "straddle": (
            ("product.strike", .005, "relative"),
        ),
        "asian_option": (
            ("product.strike", .005, "relative"),
        ),
        "athena_autocall": (
            ("product.autocall_barrier", .005, "relative"),
            ("product.protection_barrier", .005, "relative"),
            ("product.annual_coupon_rate", .0005, "absolute"),
        ),
        "cliquet": (
            ("product.participation_rate", .005, "relative"),
            ("product.local_floor", .002, "absolute"),
            ("product.local_cap", .002, "absolute"),
            ("product.global_floor", .002, "absolute"),
            ("product.global_cap", .002, "absolute"),
        ),
        "double_knock_out_option": (
            ("product.strike", .005, "relative"),
            ("product.lower_barrier", .005, "relative"),
            ("product.upper_barrier", .005, "relative"),
        ),
        "down_and_in_option": (
            ("product.strike", .005, "relative"),
            ("product.barrier", .005, "relative"),
        ),
        "down_and_out_option": (
            ("product.strike", .005, "relative"),
            ("product.barrier", .005, "relative"),
        ),
        "forward_start_option": (
            ("product.moneyness", .005, "relative"),
        ),
        "geometric_asian_option": (
            ("product.strike", .005, "relative"),
        ),
        "lookback_option": (
            ("product.strike", .005, "relative"),
        ),
        "phoenix_autocall": (
            ("product.autocall_barrier", .005, "relative"),
            ("product.coupon_barrier", .005, "relative"),
            ("product.protection_barrier", .005, "relative"),
            ("product.annual_coupon_rate", .0005, "absolute"),
        ),
        "phoenix_memory_autocall": (
            ("product.autocall_barrier", .005, "relative"),
            ("product.coupon_barrier", .005, "relative"),
            ("product.protection_barrier", .005, "relative"),
            ("product.annual_coupon_rate", .0005, "absolute"),
        ),
        "range_accrual": (
            ("product.lower_barrier", .005, "relative"),
            ("product.upper_barrier", .005, "relative"),
            ("product.coupon_rate", .0005, "absolute"),
        ),
        "up_and_in_option": (
            ("product.strike", .005, "relative"),
            ("product.barrier", .005, "relative"),
        ),
        "up_and_out_option": (
            ("product.strike", .005, "relative"),
            ("product.barrier", .005, "relative"),
        ),
        "up_no_touch": (
            ("product.barrier", .005, "relative"),
            ("product.cash_payoff", .005, "relative"),
        ),
        "up_one_touch": (
            ("product.barrier", .005, "relative"),
            ("product.cash_payoff", .005, "relative"),
        ),
        "european_swaption": (
            ("product.strike", .0005, "absolute"),
        ),
        "bermudan_swaption": (
            ("product.notional", .005, "relative"),
            ("product.strike", .0005, "absolute"),
            ("product.accrual_fraction", .005, "relative"),
        ),
        "rate_option": (
            ("product.notional", .005, "relative"),
            ("product.strike", .0005, "absolute"),
        ),
        "zero_coupon_bond_option": (
            ("product.notional", .005, "relative"),
            ("product.strike", .005, "relative"),
        ),
    }[product]
    maturities = () if product in {
        "american_option", "bermudan_swaption"
    } else (
        ("product.maturity_years", 1.0 / 504.0, "absolute"),
    )
    coordinates = (*coordinates, *curves, *products, *maturities)
    return [{"parameter": name, "displacement": h, "scale": scale,
             "boundary": "central_then_one_sided_order2"} for name, h, scale in coordinates]
