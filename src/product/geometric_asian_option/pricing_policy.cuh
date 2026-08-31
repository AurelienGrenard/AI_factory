// Geometric-average payoff composed with a dense equity schedule.
#pragma once

#include "common/compensated_sum.cuh"
#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/geometric_asian_option/parameters.hpp"

#include <cstdint>

namespace ai_factory::workbench::product {

template<OptionSide Side>
struct GeometricAsianOptionPathPolicy {
    using ProductParameters = GeometricAsianOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::log_spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        CompensatedFloatSum log_sum;
        std::uint32_t count = 0U;

        __device__ __forceinline__ bool observe(float log_spot) {
            log_sum.add(log_spot);
            ++count;
            return true;
        }

        __device__ __forceinline__ bool on_initial_value(float log_spot) {
            return observe(log_spot);
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float log_spot
        ) {
            return observe(log_spot);
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
        const float geometric_mean = expf(
            handler.log_sum.value() / static_cast<float>(handler.count)
        );
        return product.discount * payoff::vanilla_option_payoff<Side>(
            geometric_mean,
            product.strike
        );
    }
};

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::LogSpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using GeometricAsianOptionPricingPolicy =
    equity::PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        GeometricAsianOptionPathPolicy<Side>
    >;

}  // namespace ai_factory::workbench::product
