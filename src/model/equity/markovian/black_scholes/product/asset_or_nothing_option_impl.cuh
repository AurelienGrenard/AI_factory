// Shared Black-Scholes asset_or_nothing_option formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option.cuh"

#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
struct AssetOrNothingOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::AssetOrNothingOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float discounted_spot;
        float d1;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::AssetOrNothingOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years =
            time::year_fraction(product.maturity_days, time_configuration);
        const auto values = prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
        return {
            values.discounted_underlying,
            values.d1,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return asset_or_nothing_price(
            row.discounted_spot, row.d1, option_sign
        );
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    AssetOrNothingOptionClosedFormPricingPolicy<OptionSide::call>
>);


}  // namespace ai_factory::workbench::model::equity::black_scholes
