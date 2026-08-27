// Range-accrual payoff composed with a regular equity schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/range_accrual/parameters.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

struct RangeAccrualPathPolicy {
    using ProductParameters = RangeAccrualParameters;
    using Calendar = simulation::RegularCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::log_spot;

    struct PreparedProduct {
        float lower_coordinate;
        float upper_coordinate;
        float maturity_discount;
        float discounted_coupon_per_observation;
    };

    struct Handler {
        float lower_coordinate;
        float upper_coordinate;
        std::uint32_t in_range_count = 0U;

        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float log_spot
        ) {
            in_range_count += static_cast<std::uint32_t>(
                log_spot >= lower_coordinate
                && log_spot <= upper_coordinate
            );
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {
            product.observation_interval_days,
            product.maturity_days / product.observation_interval_days,
        };
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        const float discount = expf(
            -model.risk_free_rate * context.maturity_years
        );
        const float initial_log_spot = logf(model.spot);
        const float observation_years =
            static_cast<float>(product.observation_interval_days)
            * context.day_fraction;
        return {
            initial_log_spot + logf(product.lower_barrier),
            initial_log_spot + logf(product.upper_barrier),
            discount,
            discount * product.coupon_rate * observation_years,
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {product.lower_coordinate, product.upper_coordinate};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        return fmaf(
            product.discounted_coupon_per_observation,
            static_cast<float>(handler.in_range_count),
            product.maturity_discount
        );
    }
};

namespace detail {

template<equity::LogSpotDynamicsPolicy Dynamics>
struct RangeAccrualObservationHandler {
    float lower_coordinate;
    float upper_coordinate;
    std::uint32_t in_range_count = 0U;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        float coordinate = 0.0f;
        if constexpr (Dynamics::kNativeLogSpot) {
            coordinate = Dynamics::log_spot(state);
        } else {
            coordinate = Dynamics::spot(state);
        }
        in_range_count += static_cast<std::uint32_t>(
            coordinate >= lower_coordinate
            && coordinate <= upper_coordinate
        );
        return true;
    }
};

}  // namespace detail

template<
    simulation::ObservedSchedulePolicy SchedulePolicy
>
requires equity::LogSpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct RangeAccrualPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = RangeAccrualParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float lower_coordinate;
        float upper_coordinate;
        float maturity_discount;
        float discounted_coupon_per_observation;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{
            product.observation_interval_days,
            product.maturity_days / product.observation_interval_days,
        };
        const float observation_years = simulation::day_count_year_fraction(
            product.observation_interval_days,
            time_configuration
        );
        const float maturity_discount = equity::constant_rate_discount_factor(
            model,
            simulation::day_count_year_fraction(
                product.maturity_days,
                time_configuration
            )
        );
        float lower_coordinate = 0.0f;
        float upper_coordinate = 0.0f;
        if constexpr (Dynamics::kNativeLogSpot) {
            const float initial_log_spot = logf(model.spot);
            lower_coordinate = initial_log_spot + logf(product.lower_barrier);
            upper_coordinate = initial_log_spot + logf(product.upper_barrier);
        } else {
            lower_coordinate = model.spot * product.lower_barrier;
            upper_coordinate = model.spot * product.upper_barrier;
        }
        return {
            Schedule::prepare(model, calendar, time_configuration),
            lower_coordinate,
            upper_coordinate,
            maturity_discount,
            maturity_discount * product.coupon_rate * observation_years,
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        detail::RangeAccrualObservationHandler<Dynamics> handler{
            row.lower_coordinate,
            row.upper_coordinate,
        };
        Schedule::simulate(row.schedule, key, path, handler);
        return fmaf(
            row.discounted_coupon_per_observation,
            static_cast<float>(handler.in_range_count),
            row.maturity_discount
        );
    }

};

}  // namespace ai_factory::workbench::product
