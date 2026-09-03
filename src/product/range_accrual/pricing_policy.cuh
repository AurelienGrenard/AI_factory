// Range-accrual payoff composed with a regular equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/range_accrual/parameters.hpp"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::product {

struct RangeAccrualPathPolicy {
    using ProductParameters = RangeAccrualParameters;
    using Calendar = simulation::RegularCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::log_spot;

    struct PreparedProduct {
        float lower_coordinate;
        float upper_coordinate;
        float maturity_discount;
        float discounted_coupon_per_observation;
    };

    struct Handler {
        float lower_coordinate;
        float upper_coordinate;
        std::uint32_t in_range_count = 0U;

        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float log_spot
        ) {
            in_range_count += static_cast<std::uint32_t>(
                log_spot >= lower_coordinate
                && log_spot <= upper_coordinate
            );
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {
            product.observation_interval_days,
            product.maturity_days / product.observation_interval_days,
        };
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        const float discount = expf(
            -model.risk_free_rate * context.maturity_years
        );
        const float initial_log_spot = logf(model.spot);
        const float observation_years =
            static_cast<float>(product.observation_interval_days)
            * context.day_fraction;
        return {
            initial_log_spot + logf(product.lower_barrier),
            initial_log_spot + logf(product.upper_barrier),
            discount,
            discount * product.coupon_rate * observation_years,
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {product.lower_coordinate, product.upper_coordinate};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        return fmaf(
            product.discounted_coupon_per_observation,
            static_cast<float>(handler.in_range_count),
            product.maturity_discount
        );
    }
};

template<
    simulation::ObservedSchedulePolicy SchedulePolicy
>
requires equity::LogSpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using RangeAccrualPricingPolicy = equity::PathProductMonteCarloPricingPolicy<
    SchedulePolicy,
    RangeAccrualPathPolicy
>;

}  // namespace ai_factory::workbench::product
