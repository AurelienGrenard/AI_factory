// Shared Phoenix payoff policy with compile-time coupon-memory selection.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::detail {

template<equity::SpotDynamicsPolicy Dynamics, bool MemoryCoupon>
struct PhoenixObservationHandler {
    std::uint32_t observation_count;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float coupon_per_observation;
    float discount_per_observation;
    float present_value = 0.0f;
    float discount = 1.0f;
    float remembered_coupon = 0.0f;

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
        const float spot = Dynamics::spot(state);
        float coupon = 0.0f;
        if constexpr (MemoryCoupon) {
            remembered_coupon += coupon_per_observation;
            coupon = spot >= coupon_barrier ? remembered_coupon : 0.0f;
            if (coupon > 0.0f) remembered_coupon = 0.0f;
        } else {
            coupon = spot >= coupon_barrier
                ? coupon_per_observation
                : 0.0f;
        }

        const bool maturity = observation + 1U == observation_count;
        if (!maturity && spot >= autocall_barrier) {
            present_value = present_value + discount * (1.0f + coupon);
            return false;
        }
        if (maturity) {
            const float capital = spot >= protection_barrier ? 1.0f : spot;
            present_value = present_value + discount * (capital + coupon);
            return false;
        }
        present_value = fmaf(discount, coupon, present_value);
        return true;
    }
};

template<
    simulation::CountedObservedSchedulePolicy SchedulePolicy,
    typename ProductParametersT,
    bool MemoryCoupon
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct PhoenixPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float autocall_barrier;
        float coupon_barrier;
        float protection_barrier;
        float coupon_per_observation;
        float discount_per_observation;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{
            product.observation_interval,
            product.maturity / product.observation_interval,
        };
        const float observation_years = simulation::day_count_year_fraction(
            product.observation_interval,
            time_configuration
        );
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.autocall_barrier,
            product.coupon_barrier,
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
        PhoenixObservationHandler<Dynamics, MemoryCoupon> handler{
            Schedule::observation_count(row.schedule),
            row.autocall_barrier,
            row.coupon_barrier,
            row.protection_barrier,
            row.coupon_per_observation,
            row.discount_per_observation,
        };
        Schedule::simulate(row.schedule, key, path, handler);
        return handler.present_value;
    }

};

}  // namespace ai_factory::workbench::product::detail
