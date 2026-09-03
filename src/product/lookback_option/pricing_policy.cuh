// Fixed-strike lookback payoff composed with a dense equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/lookback_option/parameters.hpp"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::product {

struct LookbackOptionPathPolicy {
    using ProductParameters = LookbackOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        float maximum = 0.0f;

        __device__ __forceinline__ bool observe(float spot) {
            maximum = fmaxf(maximum, spot);
            return true;
        }

        __device__ __forceinline__ bool on_initial_value(float spot) {
            maximum = spot;
            return true;
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
        return product.discount * fmaxf(
            handler.maximum - product.strike,
            0.0f
        );
    }
};

template<
    simulation::DenseSchedulePolicy SchedulePolicy
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using LookbackOptionPricingPolicy =
    equity::PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        LookbackOptionPathPolicy
    >;

}  // namespace ai_factory::workbench::product
