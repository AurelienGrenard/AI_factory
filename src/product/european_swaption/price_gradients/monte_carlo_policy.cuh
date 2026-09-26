// European-swaption terminal node policies shared by mono and node-graph MC.
#pragma once

#include "common/fixed_income/swaption_side.cuh"
#include "common/price_gradients/time_configuration.hpp"
#include "product/european_swaption/pricing_row.cuh"
#include "product/european_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <utility>

namespace ai_factory::workbench::product::european_swaption::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<SwaptionSide Side, typename Metadata, typename State>
__device__ __forceinline__ float terminal_payoff(
    const Metadata& product,
    const State& terminal
) {
    if (!product.schedule.valid()) return ::nanf("");
    const float swap = payer_swap_value(
        product.model,
        terminal.state,
        product.exercise_time_years,
        product.exercise_time_years,
        product.strike,
        product.schedule
    );
    const float discount = discount_factor(
        product.model,
        terminal.state_integral,
        product.exercise_time_years
    );
    constexpr float sign = Side == SwaptionSide::payer ? 1.0f : -1.0f;
    return ::isfinite(swap) && ::isfinite(discount)
        ? product.notional * discount * fmaxf(sign * swap, 0.0f)
        : ::nanf("");
}

template<typename Dynamics, typename Preparation, SwaptionSide Side>
struct StandaloneTerminalNodePolicy {
    using Scenario = typename Preparation::Scenario;
    using PreparedDynamics = typename Dynamics::Prepared;
    using NodeValue = typename Dynamics::State;
    using ScheduleView = decltype(make_european_swaption_schedule_view(
        std::declval<const typename Preparation::Product&>(),
        RegularEuropeanSwaptionScheduleSource{},
        0.0f
    ));
    using Metadata = fixed_income::PreparedEuropeanSwaptionRow<
        typename Preparation::Model,
        ScheduleView
    >;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Scenario& scenario,
        pg::TimeConfiguration
    ) {
        return Dynamics::prepare(
            scenario.model,
            static_cast<float>(scenario.product.exercise_time_days)
                * scenario.day_fraction
        );
    }

    __device__ __forceinline__ static Metadata prepare_metadata(
        const Scenario& scenario,
        pg::TimeConfiguration
    ) {
        return fixed_income::prepare_european_swaption_row(
            scenario.model,
            scenario.product,
            RegularEuropeanSwaptionScheduleSource{},
            scenario.day_fraction
        );
    }

    __device__ __forceinline__ static NodeValue observe(
        const typename Dynamics::State& state
    ) {
        return state;
    }

    __device__ __forceinline__ static float payoff(
        const Metadata& metadata,
        const NodeValue& value
    ) {
        return terminal_payoff<Side>(metadata, value);
    }

    __device__ __forceinline__ static float centered_first(
        const Metadata& first,
        const Metadata& second,
        const NodeValue& first_value,
        const NodeValue& second_value,
        float represented_width
    ) {
        return (payoff(second, second_value) - payoff(first, first_value))
            / represented_width;
    }
};

template<
    typename Dynamics,
    typename Preparation,
    typename Composition,
    SwaptionSide Side
>
struct FittedTerminalNodePolicy {
    using Scenario = typename Preparation::Scenario;
    using PreparedDynamics = typename Dynamics::Prepared;
    using NodeValue = typename Dynamics::State;
    using ScheduleView = decltype(make_european_swaption_schedule_view(
        std::declval<const typename Preparation::Product&>(),
        RegularEuropeanSwaptionScheduleSource{},
        0.0f
    ));
    using Metadata = fixed_income::PreparedEuropeanSwaptionRow<
        typename Composition::FittedModel,
        ScheduleView
    >;

    __device__ __forceinline__ static PreparedDynamics prepare_dynamics(
        const Scenario& scenario,
        pg::TimeConfiguration
    ) {
        return Dynamics::prepare(
            scenario.model,
            static_cast<float>(scenario.product.exercise_time_days)
                * scenario.day_fraction
        );
    }

    __device__ __forceinline__ static Metadata prepare_metadata(
        const Scenario& scenario,
        pg::TimeConfiguration
    ) {
        return fixed_income::prepare_european_swaption_row<Composition>(
            scenario.model,
            scenario.curve,
            scenario.product,
            RegularEuropeanSwaptionScheduleSource{},
            scenario.day_fraction
        );
    }

    __device__ __forceinline__ static NodeValue observe(
        const typename Dynamics::State& state
    ) {
        return state;
    }

    __device__ __forceinline__ static float payoff(
        const Metadata& metadata,
        const NodeValue& value
    ) {
        return terminal_payoff<Side>(metadata, value);
    }

    __device__ __forceinline__ static float centered_first(
        const Metadata& first,
        const Metadata& second,
        const NodeValue& first_value,
        const NodeValue& second_value,
        float represented_width
    ) {
        return (payoff(second, second_value) - payoff(first, first_value))
            / represented_width;
    }
};

template<SwaptionSide Side>
struct StandaloneMonteCarloPolicy {
    template<typename Dynamics, typename Preparation>
    using SensitivityNodePolicy = StandaloneTerminalNodePolicy<
        Dynamics, Preparation, Side
    >;
};

template<typename Composition, SwaptionSide Side>
struct FittedMonteCarloPolicy {
    template<typename Dynamics, typename Preparation>
    using SensitivityNodePolicy = FittedTerminalNodePolicy<
        Dynamics, Preparation, Composition, Side
    >;
};

}  // namespace ai_factory::workbench::product::european_swaption::price_gradients
