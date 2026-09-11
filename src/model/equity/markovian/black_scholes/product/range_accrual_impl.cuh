// Shared Black-Scholes range_accrual formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/range_accrual.cuh"

#include "common/compensated_sum.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

struct RangeAccrualClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::RangeAccrualParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        LognormalEvolutionContext evolution;
        float log_lower_barrier;
        float log_upper_barrier;
        float observation_interval_years;
        float maturity_discount;
        float discounted_coupon_per_observation;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::RangeAccrualParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float observation_interval_years = time::year_fraction(
            product.observation_interval_days,
            time_configuration
        );
        const float maturity_years = time::year_fraction(
            product.maturity_days,
            time_configuration
        );
        const float maturity_discount =
            expf(-model.risk_free_rate * maturity_years);
        const float log_initial_spot = logf(model.spot);
        return {
            prepare_lognormal_evolution(prepare_analytics(model)),
            log_initial_spot + logf(product.lower_barrier),
            log_initial_spot + logf(product.upper_barrier),
            observation_interval_years,
            maturity_discount,
            maturity_discount * product.coupon_rate
                * observation_interval_years,
            product.maturity_days / product.observation_interval_days,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        CompensatedFloatSum probability_sum;
        for (std::uint32_t observation = 1U;
             observation <= row.observation_count;
             ++observation) {
            const float observation_time =
                static_cast<float>(observation)
                * row.observation_interval_years;
            probability_sum.add(lognormal_log_interval_probability(
                row.evolution,
                row.log_lower_barrier,
                row.log_upper_barrier,
                observation_time
            ));
        }
        return fmaf(
            row.discounted_coupon_per_observation,
            probability_sum.value(),
            row.maturity_discount
        );
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    RangeAccrualClosedFormPricingPolicy
>);


}  // namespace ai_factory::workbench::model::equity::black_scholes
