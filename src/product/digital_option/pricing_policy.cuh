// Cash-or-nothing payoff composed with a terminal equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/option_side.cuh"
#include "product/digital_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct DigitalOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = DigitalOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float strike;
        float discounted_cash_payoff;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{product.maturity};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.strike,
            product.cash_payoff * equity::constant_rate_discount_factor(
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
        const bool pays = Side == OptionSide::call
            ? terminal_spot > row.strike
            : terminal_spot < row.strike;
        return pays ? row.discounted_cash_payoff : 0.0f;
    }

};

}  // namespace ai_factory::workbench::product
