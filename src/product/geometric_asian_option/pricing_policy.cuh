// Geometric-average payoff composed with a dense equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/geometric_asian_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::LogSpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct GeometricAsianOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = GeometricAsianOptionParameters;
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
        equity::GeometricMeanObservationHandler<Dynamics> handler;
        Schedule::simulate(row.schedule, key, path, handler);
        return row.discount * payoff::vanilla_option_payoff<Side>(
            handler.geometric_mean(
                Schedule::observation_count(row.schedule) + 1U
            ),
            row.strike
        );
    }

};

}  // namespace ai_factory::workbench::product
