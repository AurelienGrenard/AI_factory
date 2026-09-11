// Shared Black-Scholes digital_option formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/digital_option.cuh"

#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
struct DigitalOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::DigitalOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float discounted_cash_payoff;
        float d2;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::DigitalOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years =
            time::year_fraction(product.maturity_days, time_configuration);
        const auto values = prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
        return {
            product.cash_payoff
                * expf(-model.risk_free_rate * maturity_years),
            values.d2,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return cash_or_nothing_price(
            row.discounted_cash_payoff, row.d2, option_sign
        );
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    DigitalOptionClosedFormPricingPolicy<OptionSide::call>
>);


}  // namespace ai_factory::workbench::model::equity::black_scholes
