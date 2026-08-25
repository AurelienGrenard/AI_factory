// Double-knock-out payoff composed with a dense equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/simulation/barrier_handlers.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/observables.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/double_knock_out_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct DoubleKnockOutOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = DoubleKnockOutOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float strike;
        float lower_barrier;
        float upper_barrier;
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
            product.lower_barrier,
            product.upper_barrier,
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
        simulation::DoubleKnockOutObservationHandler<
            Dynamics,
            equity::SpotObservable<Dynamics>
        > handler{
            row.lower_barrier,
            row.upper_barrier,
        };
        Schedule::simulate(row.schedule, key, path, handler);
        if (handler.terminal_value <= row.lower_barrier
            || handler.terminal_value >= row.upper_barrier) {
            return 0.0f;
        }
        return row.discount * payoff::vanilla_option_payoff<Side>(
            handler.terminal_value,
            row.strike
        );
    }

};

}  // namespace ai_factory::workbench::product
