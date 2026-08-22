// Up-and-out payoff composed with a densely monitored equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/option_side.cuh"
#include "product/up_and_out_option/dataset.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {
namespace detail {

template<equity::EquityDynamicsPolicy Dynamics>
struct UpAndOutObservationHandler {
    float barrier;
    float terminal_spot = 0.0f;

    __device__ __forceinline__ bool observe(
        const typename Dynamics::State& state
    ) {
        terminal_spot = Dynamics::spot(state);
        return !(terminal_spot >= barrier);
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return observe(state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        return observe(state);
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
struct UpAndOutOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = UpAndOutOptionParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float strike;
        float barrier;
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
        const typename Schedule::Definition definition{product.maturity};
        return {
            Schedule::prepare(
                model,
                definition,
                configuration,
                inputs.schedule
            ),
            philox::make_key(seed),
            product.strike,
            product.barrier,
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
        detail::UpAndOutObservationHandler<Dynamics> handler{row.barrier};
        Schedule::simulate(row.schedule, row.key, path, handler);
        if (handler.terminal_spot >= row.barrier) return 0.0f;
        if constexpr (Side == OptionSide::call) {
            return row.discount
                * fmaxf(handler.terminal_spot - row.strike, 0.0f);
        } else {
            return row.discount
                * fmaxf(row.strike - handler.terminal_spot, 0.0f);
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
