// Shared Black-Scholes straddle formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/straddle.cuh"

#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

struct StraddleClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::StraddleParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::StraddleParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years = time::year_fraction(
            product.maturity_days,
            time_configuration
        );
        return prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        return discounted_lognormal_option_price(row, 1.0f)
            + discounted_lognormal_option_price(row, -1.0f);
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    StraddleClosedFormPricingPolicy
>);


}  // namespace ai_factory::workbench::model::equity::black_scholes
