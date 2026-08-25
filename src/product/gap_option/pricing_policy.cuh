// Gap-option payoff composed with a terminal equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/option_side.cuh"
#include "product/gap_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct GapOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = GapOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float trigger_strike;
        float payoff_strike;
        float discount;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{product.maturity};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.trigger_strike,
            product.payoff_strike,
            equity::constant_rate_discount_factor(
                model,
                simulation::day_count_year_fraction(
                    product.maturity,
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
        const typename Dynamics::State terminal = Schedule::simulate_terminal(
            row.schedule,
            key,
            path
        );
        const float terminal_spot = Dynamics::spot(terminal);
        if constexpr (Side == OptionSide::call) {
            return terminal_spot > row.trigger_strike
                ? row.discount * (terminal_spot - row.payoff_strike)
                : 0.0f;
        } else {
            return terminal_spot < row.trigger_strike
                ? row.discount * (row.payoff_strike - terminal_spot)
                : 0.0f;
        }
    }

};

}  // namespace ai_factory::workbench::product
