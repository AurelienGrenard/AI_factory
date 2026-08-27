// Athena payoff composed with arbitrary equity schedule and discount policies.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/athena_autocall/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

struct AthenaAutocallPathPolicy {
    using ProductParameters = AthenaAutocallParameters;
    using Calendar = simulation::RegularCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        std::uint32_t observation_count;
        float autocall_barrier;
        float protection_barrier;
        float gain_per_observation;
        float discount_per_observation;
    };

    struct Handler {
        std::uint32_t observation_count;
        float autocall_barrier;
        float protection_barrier;
        float gain_per_observation;
        float discount_per_observation;
        float discount = 1.0f;
        float accumulated_gain = 0.0f;
        float discounted_payoff = 0.0f;

        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            float spot
        ) {
            discount *= discount_per_observation;
            accumulated_gain += gain_per_observation;
            const bool maturity = observation + 1U == observation_count;
            if (!maturity && spot >= autocall_barrier) {
                discounted_payoff = discount * (1.0f + accumulated_gain);
                return false;
            }
            if (maturity) {
                const float redemption = spot >= autocall_barrier
                    ? 1.0f + accumulated_gain
                    : (spot >= protection_barrier ? 1.0f : spot);
                discounted_payoff = discount * redemption;
                return false;
            }
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
        const std::uint32_t observation_count =
            product.maturity_days / product.observation_interval_days;
        const float observation_years =
            static_cast<float>(product.observation_interval_days)
            * context.day_fraction;
        return {
            observation_count,
            product.autocall_barrier,
            product.protection_barrier,
            product.annual_coupon_rate * observation_years,
            expf(-model.risk_free_rate * observation_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {
            product.observation_count,
            product.autocall_barrier,
            product.protection_barrier,
            product.gain_per_observation,
            product.discount_per_observation,
        };
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct&,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        return handler.discounted_payoff;
    }
};

namespace detail {

template<equity::SpotDynamicsPolicy Dynamics>
struct AthenaAutocallObservationHandler {
    std::uint32_t observation_count;
    float autocall_barrier;
    float protection_barrier;
    float gain_per_observation;
    float discount_per_observation;
    float discount = 1.0f;
    float accumulated_gain = 0.0f;
    float discounted_payoff = 0.0f;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        discount *= discount_per_observation;
        accumulated_gain += gain_per_observation;
        const float current_spot = Dynamics::spot(state);
        const bool maturity = observation + 1U == observation_count;

        if (!maturity && current_spot >= autocall_barrier) {
            discounted_payoff = discount * (1.0f + accumulated_gain);
            return false;
        }
        if (maturity) {
            const float redemption =
                current_spot >= autocall_barrier
                ? 1.0f + accumulated_gain
                : (
                    current_spot >= protection_barrier
                    ? 1.0f
                    : current_spot
                );
            discounted_payoff = discount * redemption;
            return false;
        }
        return true;
    }
};

}  // namespace detail

template<
    simulation::CountedObservedSchedulePolicy SchedulePolicy
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct AthenaAutocallPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = AthenaAutocallParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float autocall_barrier;
        float protection_barrier;
        float gain_per_observation;
        float discount_per_observation;
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
        const float observation_years = simulation::day_count_year_fraction(
            product.observation_interval_days,
            time_configuration
        );
        return {
            Schedule::prepare(
                model,
                calendar,
                time_configuration
            ),
            product.autocall_barrier,
            product.protection_barrier,
            product.annual_coupon_rate * observation_years,
            equity::constant_rate_discount_factor(
                model,
                observation_years
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        detail::AthenaAutocallObservationHandler<Dynamics> handler{
            Schedule::observation_count(row.schedule),
            row.autocall_barrier,
            row.protection_barrier,
            row.gain_per_observation,
            row.discount_per_observation,
        };
        Schedule::simulate(
            row.schedule,
            key,
            path,
            handler
        );
        return handler.discounted_payoff;
    }

};

}  // namespace ai_factory::workbench::product
