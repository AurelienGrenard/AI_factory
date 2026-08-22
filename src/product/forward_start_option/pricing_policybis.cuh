// Two-date forward-start payoff composed with an exact calendar schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/option_side.cuh"
#include "product/forward_start_option/dataset.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {
namespace detail {

template<equity::EquityDynamicsPolicy Dynamics>
struct ForwardStartObservationHandler {
    float reset_spot = 0.0f;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        if (observation == 0U) {
            reset_spot = Dynamics::spot(state);
            return true;
        }
        return false;
    }
};

}  // namespace detail

template<
    equity::EquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    OptionSide Side
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct ForwardStartOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ForwardStartOptionParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    static_assert(Schedule::kObservationCount == 2U);

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float moneyness;
        float discount;
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
        const typename Schedule::Definition definition{{
            product.reset_time,
            product.maturity - product.reset_time,
        }};
        return {
            Schedule::prepare(
                model,
                definition,
                configuration,
                inputs.schedule
            ),
            philox::make_key(seed),
            product.moneyness,
            Discount::discount_factor(
                model,
                inputs.discount,
                Schedule::total_year_fraction(definition, configuration)
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        detail::ForwardStartObservationHandler<Dynamics> handler;
        const typename Dynamics::State terminal = Schedule::simulate(
            row.schedule,
            row.key,
            path,
            handler
        );
        const float terminal_spot = Dynamics::spot(terminal);
        const float strike = row.moneyness * handler.reset_spot;
        if constexpr (Side == OptionSide::call) {
            return row.discount
                * fmaxf(terminal_spot - strike, 0.0f);
        } else {
            return row.discount
                * fmaxf(strike - terminal_spot, 0.0f);
        }
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
