// Arithmetic-average payoff composed with dense equity schedules.
#pragma once

#include "common/compensated_sum.cuh"
#include "common/equity/concepts.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/asian_option/parameters.hpp"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::product {

template<OptionSide Side>
struct AsianOptionPathPolicy {
    using ProductParameters = AsianOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        CompensatedFloatSum sum;
        std::uint32_t count = 0U;

        __device__ __forceinline__ bool observe(float spot) {
            sum.add(spot);
            ++count;
            return true;
        }

        __device__ __forceinline__ bool on_initial_value(float spot) {
            return observe(spot);
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float spot
        ) {
            return observe(spot);
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {product.maturity_days};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.strike,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        const float mean =
            handler.sum.value() / static_cast<float>(handler.count);
        return product.discount
            * payoff::vanilla_option_payoff<Side>(mean, product.strike);
    }
};

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using AsianOptionPricingPolicy = equity::PathProductMonteCarloPricingPolicy<
    SchedulePolicy,
    AsianOptionPathPolicy<Side>
>;

}  // namespace ai_factory::workbench::product
