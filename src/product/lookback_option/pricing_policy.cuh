// Fixed-strike lookback payoff composed with a dense equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/equity/handlers.cuh"
#include "product/lookback_option/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

struct LookbackOptionPathPolicy {
    using ProductParameters = LookbackOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        float maximum = 0.0f;

        __device__ __forceinline__ bool observe(float spot) {
            maximum = fmaxf(maximum, spot);
            return true;
        }

        __device__ __forceinline__ bool on_initial_value(float spot) {
            maximum = spot;
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float spot
        ) {
            return observe(spot);
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
        return product.discount * fmaxf(
            handler.maximum - product.strike,
            0.0f
        );
    }
};

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
        equity::MaximumObservationHandler<Dynamics> handler;
        Schedule::simulate(row.schedule, key, path, handler);
        return row.discount * fmaxf(handler.maximum() - row.strike, 0.0f);
    }

};

}  // namespace ai_factory::workbench::product
