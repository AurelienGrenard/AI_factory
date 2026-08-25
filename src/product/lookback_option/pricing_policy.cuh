// Fixed-strike lookback payoff composed with a dense equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "product/lookback_option/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    simulation::DenseSchedulePolicy SchedulePolicy
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct LookbackOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = LookbackOptionParameters;
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
        equity::MaximumObservationHandler<Dynamics> handler;
        Schedule::simulate(row.schedule, key, path, handler);
        return row.discount * fmaxf(handler.maximum() - row.strike, 0.0f);
    }

};

}  // namespace ai_factory::workbench::product
