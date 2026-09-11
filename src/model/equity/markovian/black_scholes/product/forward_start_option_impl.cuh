// Shared Black-Scholes forward_start_option formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/forward_start_option.cuh"

#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
struct ForwardStartOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::ForwardStartOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::ForwardStartOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float reset_time_years =
            time::year_fraction(product.reset_time_days, time_configuration);
        const float maturity_years =
            time::year_fraction(product.maturity_days, time_configuration);
        return prepare_forward_start_option_values(
            prepare_analytics(model),
            product.moneyness,
            reset_time_years,
            maturity_years
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return discounted_lognormal_option_price(row, option_sign);
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    ForwardStartOptionClosedFormPricingPolicy<OptionSide::call>
>);


}  // namespace ai_factory::workbench::model::equity::black_scholes
