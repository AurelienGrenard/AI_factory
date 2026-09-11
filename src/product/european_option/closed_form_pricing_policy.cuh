// European payoff composition for models exposing discounted lognormal analytics.
#pragma once

#include "common/device_inputs.cuh"
#include "common/lognormal_option.cuh"
#include "common/option_side.cuh"
#include "common/time_configuration.cuh"
#include "product/european_option/parameters.hpp"

namespace ai_factory::workbench::product {

template<typename Model>
concept LognormalEuropeanAnalytics = requires(const Model& model, float strike, float years) {
    { prepare_vanilla_option_values(prepare_analytics(model), strike, years) }
        -> std::same_as<DiscountedLognormalOptionValues>;
};

template<LognormalEuropeanAnalytics Model, OptionSide Side>
struct LognormalEuropeanOptionClosedFormPolicy {
    using DeviceInputs = ModelProductDeviceInputs<Model, EuropeanOptionParameters>;
    using TimeConfiguration = time::DayFractionTimeConfiguration;
    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model, const EuropeanOptionParameters& product,
        const TimeConfiguration& configuration
    ) {
        const float maturity_years = time::year_fraction(product.maturity_days, configuration);
        return prepare_vanilla_option_values(prepare_analytics(model), product.strike, maturity_years);
    }
    __device__ __forceinline__ static float evaluate_price(const PreparedRow& row) {
        constexpr float option_sign = Side == OptionSide::call ? 1.0f : -1.0f;
        return discounted_lognormal_option_price(row, option_sign);
    }
};

}  // namespace ai_factory::workbench::product
