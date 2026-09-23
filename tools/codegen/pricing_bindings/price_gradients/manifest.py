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
            "device_prepared_step_terminal",
            "device_prepared_exact_terminal",
            "device_prepared_lsm",
            "device_prepared_fixed_income_lsm",
            "host_prepared_lsm",
            "device_prepared_cooperative_closed_form",
        }
        if self.preparation_strategy not in supported:
            raise ValueError(
                "Unknown price-gradient preparation strategy: "
                + self.preparation_strategy
            )
        if self.supported_orders not in {
            ("first",),
            ("first", "diagonal_second"),
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
    equity = tuple(
        PriceGradientBindingSpec(binding,
                                 model_parameter_count[binding.model] + 1
                                 if binding.product == "american_option"
                                 else model_parameter_count[binding.model]
                                 + TERMINAL_PRODUCT_METADATA[
                                     binding.product
                                 ]["parameter_count"]
                                 + supports_maturity[binding.model],
                                 binding.model == "black_scholes",
                                 "device_prepared_lsm"
                                 if binding.product == "american_option"
                                 else terminal_strategy[binding.model],
                                 ("first", "diagonal_second"))
        for binding in pricing_bindings
        if binding.model in model_parameter_count and (
            binding.product in TERMINAL_PRODUCT_METADATA
            or (binding.model in lsm_models
                and binding.product == "american_option")
        ) and not (
            binding.model == "black_scholes"
            and binding.product not in {
                "european_option", "american_option"
            }
        )
    )
    fixed_income_closed_form = tuple(
        PriceGradientBindingSpec(
            binding,
            7,
            True,
            "device_prepared_cooperative_closed_form",
            ("first", "diagonal_second"),
        )
        for binding in pricing_bindings
        if binding.asset_class == "fixed_income"
        and binding.model == "cir"
        and binding.curve is None
        and binding.product == "european_swaption"
        and binding.engine == "fixed_income_closed_form"
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
        "nelson_siegel": 4,
        "svensson": 6,
    }
    fixed_income_lsm = tuple(
        PriceGradientBindingSpec(
            binding,
            fixed_income_model_parameter_count[binding.model]
            + curve_parameter_count[binding.curve]
            + 3,
            False,
            "device_prepared_fixed_income_lsm",
            ("first", "diagonal_second"),
        )
        for binding in pricing_bindings
        if binding.asset_class == "fixed_income"
        and binding.product == "bermudan_swaption"
        and binding.engine == "fixed_income_lsm"
    )
    return (*equity, *fixed_income_closed_form, *fixed_income_lsm)


def compose_datasets(delta_datasets, price_datasets, bindings):
    supported = {(spec.pricing.model, spec.pricing.product) for spec in bindings}
    diagonal_pairs = {
        (spec.pricing.model, spec.pricing.product) for spec in bindings
        if "diagonal_second" in spec.supported_orders
        and spec.preparation_strategy.startswith("device_prepared_")
    }
    equity = tuple(replace(dataset,
        dataset_id=dataset.dataset_id.removesuffix("_price_delta") + "_price_gradients",
        dataset_kind="price_gradients",
        generator_path=dataset.generator_path.replace("/price_delta/", "/price_gradients/")
            .replace("_price_delta/", "_price_gradients/"),
        template="catalog/pricing/price_gradients/generator.cpp.tpl",
        numerical_profile="selected_gradients_production_paths", layout="row_major_selected_gradients",
        sensitivity_orders=("first",))
        for dataset in delta_datasets
        if (dataset.model, dataset.product) in supported)
    fixed_income = tuple(replace(dataset,
        dataset_id=dataset.dataset_id + "_price_gradients",
        dataset_kind="price_gradients",
        generator_path=dataset.generator_path.replace("/prices/", "/price_gradients/")
            .replace("/" + dataset.dataset_id + "/",
                     "/" + dataset.dataset_id + "_price_gradients/"),
        template=(
            "catalog/pricing/price_gradients/"
            "fixed_income_bermudan_swaption_generator.cpp.tpl"
            if dataset.product == "bermudan_swaption"
            else "catalog/pricing/price_gradients/"
            "fixed_income_european_swaption_generator.cpp.tpl"
        ),
        numerical_profile=(
            "selected_gradients_production_paths"
            if dataset.product == "bermudan_swaption"
            else "selected_gradients_closed_form"
        ),
        layout="row_major_selected_gradients", sensitivity_orders=("first",))
        for dataset in price_datasets
        if dataset.dataset_kind == "prices"
        and dataset.asset_class == "fixed_income"
        and dataset.product in {"european_swaption", "bermudan_swaption"}
        and (dataset.model, dataset.product) in supported
        and any(spec.pricing.model == dataset.model
                and spec.pricing.product == dataset.product
                and spec.pricing.curve == dataset.curve
                for spec in bindings))
    diagonal = tuple(replace(dataset,
        dataset_id=dataset.dataset_id + "_diagonal",
        generator_path=dataset.generator_path.replace(
            "/" + dataset.dataset_id + "/",
            "/" + dataset.dataset_id + "_diagonal/"),
        template=("catalog/pricing/price_gradients/fixed_income_bermudan_swaption_diagonal_generator.cpp.tpl"
                  if dataset.product == "bermudan_swaption"
                  else "catalog/pricing/price_gradients/fixed_income_european_swaption_diagonal_generator.cpp.tpl"
                  if dataset.asset_class == "fixed_income"
                  else "catalog/pricing/price_gradients/american_option_diagonal_generator.cpp.tpl"
                  if dataset.product == "american_option"
                  else "catalog/pricing/price_gradients/terminal_diagonal_generator.cpp.tpl"),
        numerical_profile="selected_first_and_diagonal_second_production_paths",
        sensitivity_orders=("first", "diagonal_second"))
        for dataset in (*equity, *fixed_income)
        if (dataset.model, dataset.product) in diagonal_pairs)
    return (*equity, *fixed_income, *diagonal)


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
        "european_swaption": (
            ("product.strike", .0005, "absolute"),
        ),
        "bermudan_swaption": (
            ("product.notional", .005, "relative"),
            ("product.strike", .0005, "absolute"),
            ("product.accrual_fraction", .005, "relative"),
        ),
    }[product]
    coordinates = (*coordinates, *curves, *products)
    return [{"parameter": name, "displacement": h, "scale": scale,
             "boundary": "central_then_one_sided_order2"} for name, h, scale in coordinates]
