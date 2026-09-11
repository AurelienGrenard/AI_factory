// Shared Black-Scholes geometric_asian_option formula policy used by price and price-delta kernels.
#pragma once

#include "common/closed_form/concepts.cuh"

#include "model/equity/markovian/black_scholes/product/geometric_asian_option.cuh"

#include "common/device_inputs.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

#include <stdexcept>

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
struct GeometricAsianOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::GeometricAsianOptionParameters
    >;
    using TimeConfiguration = simulation::FixedStepTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::GeometricAsianOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const std::uint32_t transition_count =
            time_configuration.simulation_steps_per_day * product.maturity_days;
        const float maturity_years =
            static_cast<float>(transition_count) * time_configuration.dt;
        return prepare_geometric_asian_option_values(
            prepare_analytics(model),
            product.strike,
            maturity_years,
            transition_count
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
    GeometricAsianOptionClosedFormPricingPolicy<OptionSide::call>
>);


inline void validate_geometric_asian_calendar(
    const product::GeometricAsianOptionParameters* host_products,
    std::size_t product_count, PriceConstruction construction, std::size_t result_count,
    const simulation::FixedStepTimeConfiguration& time_configuration
) {
    if (host_products == nullptr || product_count == 0U) {
        throw std::invalid_argument(
            "Geometric-Asian host products must be non-empty."
        );
    }
    if (construction == PriceConstruction::Aligned
        && product_count != result_count) {
        throw std::invalid_argument(
            "Aligned Geometric-Asian host products must match results."
        );
    }
    for (std::size_t product_index = 0U;
         product_index < product_count;
         ++product_index) {
        simulation::validate_calendar(
            simulation::MaturityCalendar{
                host_products[product_index].maturity_days
            },
            time_configuration
        );
    }
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
