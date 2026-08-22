// Cliquet payoff composed with a regular equity schedule.
#pragma once

#include "common/equity/discount.cuh"
#include "product/cliquet/dataset.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {
namespace detail {

template<equity::EquitySchedulePolicy Schedule>
struct CliquetObservationHandler {
    using Dynamics = typename Schedule::Dynamics;

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
    equity::EquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct CliquetPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = CliquetParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float participation_rate;
        float local_floor;
        float local_cap;
        float global_floor;
        float global_cap;
        float maturity_discount;
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
        return {
            Schedule::prepare(model, definition, configuration, inputs.schedule),
            philox::make_key(seed),
            product.participation_rate,
            product.local_floor,
            product.local_cap,
            product.global_floor,
            product.global_cap,
            Discount::discount_factor(
                model,
                inputs.discount,
                equity::day_count_year_fraction(
                    product.maturity,
                    configuration
                )
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        detail::CliquetObservationHandler<Schedule> handler{
            row.participation_rate,
            row.local_floor,
            row.local_cap,
        };
        Schedule::simulate(row.schedule, row.key, path, handler);
        const float final_return = fminf(
            row.global_cap,
            fmaxf(row.global_floor, handler.accumulated_return)
        );
        return row.maturity_discount * (1.0f + final_return);
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
