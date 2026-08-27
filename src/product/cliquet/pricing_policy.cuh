// Cliquet payoff composed with a regular equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/cliquet/parameters.hpp"

#include <cmath>
#include <cstddef>
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

namespace detail {

template<equity::SpotDynamicsPolicy Dynamics>
struct CliquetObservationHandler {
    float participation_rate;
    float local_floor;
    float local_cap;
    float previous_spot = 0.0f;
    float accumulated_return = 0.0f;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        previous_spot = Dynamics::spot(state);
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        const float spot = Dynamics::spot(state);
        const float participated_return = participation_rate
            * (spot / previous_spot - 1.0f);
        const float local_return = fminf(
            local_cap,
            fmaxf(local_floor, participated_return)
        );
        accumulated_return += local_return;
        previous_spot = spot;
        return true;
    }
};

}  // namespace detail

template<
    simulation::ObservedSchedulePolicy SchedulePolicy
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct CliquetPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = CliquetParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float participation_rate;
        float local_floor;
        float local_cap;
        float global_floor;
        float global_cap;
        float maturity_discount;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{
            product.observation_interval_days,
            product.maturity_days / product.observation_interval_days,
        };
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.participation_rate,
            product.local_floor,
            product.local_cap,
            product.global_floor,
            product.global_cap,
            equity::constant_rate_discount_factor(
                model,
                simulation::day_count_year_fraction(
                    product.maturity_days,
                    time_configuration
                )
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        detail::CliquetObservationHandler<Dynamics> handler{
            row.participation_rate,
            row.local_floor,
            row.local_cap,
        };
        Schedule::simulate(row.schedule, key, path, handler);
        const float final_return = fminf(
            row.global_cap,
            fmaxf(row.global_floor, handler.accumulated_return)
        );
        return row.maturity_discount * (1.0f + final_return);
    }

};

}  // namespace ai_factory::workbench::product
