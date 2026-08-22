// Range-accrual payoff composed with a regular equity schedule.
#pragma once

#include "common/equity/discount.cuh"
#include "product/range_accrual/dataset.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {
namespace detail {

template<equity::EquitySchedulePolicy Schedule>
struct RangeAccrualObservationHandler {
    using Dynamics = typename Schedule::Dynamics;

    float lower_coordinate;
    float upper_coordinate;
    std::uint32_t in_range_count = 0U;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        float coordinate = 0.0f;
        if constexpr (Dynamics::kNativeLogSpot) {
            coordinate = Dynamics::log_spot(state);
        } else {
            coordinate = Dynamics::spot(state);
        }
        in_range_count += static_cast<std::uint32_t>(
            coordinate >= lower_coordinate
            && coordinate <= upper_coordinate
        );
        return true;
    }
};

}  // namespace detail

template<
    equity::EquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
> && requires {
    {
        SchedulePolicy::Dynamics::kNativeLogSpot
    } -> std::convertible_to<bool>;
}
struct RangeAccrualPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = RangeAccrualParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float lower_coordinate;
        float upper_coordinate;
        float maturity_discount;
        float discounted_coupon_per_observation;
    };

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
        const float maturity_discount = Discount::discount_factor(
            model,
            inputs.discount,
            equity::day_count_year_fraction(
                product.maturity,
                configuration
            )
        );
        float lower_coordinate = 0.0f;
        float upper_coordinate = 0.0f;
        if constexpr (Dynamics::kNativeLogSpot) {
            const float initial_log_spot = logf(model.spot);
            lower_coordinate = initial_log_spot + logf(product.lower_barrier);
            upper_coordinate = initial_log_spot + logf(product.upper_barrier);
        } else {
            lower_coordinate = model.spot * product.lower_barrier;
            upper_coordinate = model.spot * product.upper_barrier;
        }
        return {
            Schedule::prepare(model, definition, configuration, inputs.schedule),
            philox::make_key(seed),
            lower_coordinate,
            upper_coordinate,
            maturity_discount,
            maturity_discount * product.coupon_rate * observation_years,
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        detail::RangeAccrualObservationHandler<Schedule> handler{
            row.lower_coordinate,
            row.upper_coordinate,
        };
        Schedule::simulate(row.schedule, row.key, path, handler);
        return fmaf(
            row.discounted_coupon_per_observation,
            static_cast<float>(handler.in_range_count),
            row.maturity_discount
        );
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

}  // namespace ai_factory::workbench::product
