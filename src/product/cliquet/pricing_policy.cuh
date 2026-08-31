// Cliquet payoff composed with a regular equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/cliquet/parameters.hpp"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::product {

struct CliquetPathPolicy {
    using ProductParameters = CliquetParameters;
    using Calendar = simulation::RegularCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float participation_rate;
        float local_floor;
        float local_cap;
        float global_floor;
        float global_cap;
        float maturity_discount;
    };

    struct Handler {
        float participation_rate;
        float local_floor;
        float local_cap;
        float previous_spot = 0.0f;
        float accumulated_return = 0.0f;

        __device__ __forceinline__ bool on_initial_value(float spot) {
            previous_spot = spot;
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float spot
        ) {
            const float participated_return = participation_rate
                * (spot / previous_spot - 1.0f);
            accumulated_return += fminf(
                local_cap,
                fmaxf(local_floor, participated_return)
            );
            previous_spot = spot;
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
        return {
            product.participation_rate,
            product.local_floor,
            product.local_cap,
            product.global_floor,
            product.global_cap,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {
            product.participation_rate,
            product.local_floor,
            product.local_cap,
        };
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        const float final_return = fminf(
            product.global_cap,
            fmaxf(product.global_floor, handler.accumulated_return)
        );
        return product.maturity_discount * (1.0f + final_return);
    }
};

template<
    simulation::ObservedSchedulePolicy SchedulePolicy
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using CliquetPricingPolicy = equity::PathProductMonteCarloPricingPolicy<
    SchedulePolicy,
    CliquetPathPolicy
>;

}  // namespace ai_factory::workbench::product
