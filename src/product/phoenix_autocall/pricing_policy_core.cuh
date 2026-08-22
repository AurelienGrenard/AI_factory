// Shared Phoenix payoff policy with compile-time coupon-memory selection.
#pragma once

#include "common/equity/discount.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product::detail {

template<equity::EquitySchedulePolicy Schedule, bool MemoryCoupon>
struct PhoenixObservationHandler {
    using Dynamics = typename Schedule::Dynamics;

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
    equity::EquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    typename ProductParametersT,
    bool MemoryCoupon
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct PhoenixPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float autocall_barrier;
        float coupon_barrier;
        float protection_barrier;
        float coupon_per_observation;
        float discount_per_observation;
    };

    static_assert(std::is_trivially_copyable_v<ProductParameters>);
    static_assert(std::is_trivially_copyable_v<DeviceInputs>);
    static_assert(std::is_trivially_copyable_v<PreparedRow>);

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::uint64_t seed
    ) {
        const typename Schedule::Definition definition{
            product.observation_interval,
            product.maturity / product.observation_interval,
        };
        const float observation_years = Schedule::interval_year_fraction(
            definition,
            0U,
            configuration
        );
        return {
            Schedule::prepare(model, definition, configuration, inputs.schedule),
            philox::make_key(seed),
            product.autocall_barrier,
            product.coupon_barrier,
            product.protection_barrier,
            product.annual_coupon_rate * observation_years,
            Discount::discount_factor(
                model,
                inputs.discount,
                observation_years
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        PhoenixObservationHandler<Schedule, MemoryCoupon> handler{
            Schedule::observation_count(row.schedule),
            row.autocall_barrier,
            row.coupon_barrier,
            row.protection_barrier,
            row.coupon_per_observation,
            row.discount_per_observation,
        };
        Schedule::simulate(row.schedule, row.key, path, handler);
        return handler.present_value;
    }

    static void validate_configuration(
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::size_t monte_carlo_paths_per_price
    ) {
        Schedule::validate_configuration(
            configuration,
            inputs.schedule,
            monte_carlo_paths_per_price
        );
        Discount::validate_inputs(inputs.discount);
    }
};

}  // namespace ai_factory::workbench::product::detail
