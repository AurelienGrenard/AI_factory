// Geometric-average payoff composed with a dense equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/geometric_asian_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<OptionSide Side>
struct GeometricAsianOptionPathPolicy {
    using ProductParameters = GeometricAsianOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::log_spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        double log_sum = 0.0;
        std::uint32_t count = 0U;

        __device__ __forceinline__ bool observe(float log_spot) {
            log_sum += static_cast<double>(log_spot);
            ++count;
            return true;
        }

        __device__ __forceinline__ bool on_initial_value(float log_spot) {
            return observe(log_spot);
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float log_spot
        ) {
            return observe(log_spot);
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {product.maturity_days};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.strike,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        const float geometric_mean = expf(static_cast<float>(
            handler.log_sum / static_cast<double>(handler.count)
        ));
        return product.discount * payoff::vanilla_option_payoff<Side>(
            geometric_mean,
            product.strike
        );
    }
};

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
        const typename Schedule::Calendar calendar{product.maturity_days};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.strike,
            equity::constant_rate_discount_factor(
                model,
                simulation::day_count_year_fraction(
                    product.maturity_days,
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
