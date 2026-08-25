// European-option payoff composed with a terminal equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/european_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct EuropeanOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = EuropeanOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float strike;
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
            product.strike,
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
        return row.discount
            * payoff::vanilla_option_payoff<Side>(Dynamics::spot(terminal), row.strike);
    }

};

}  // namespace ai_factory::workbench::product
