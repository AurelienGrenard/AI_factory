// Two-date forward-start payoff composed with an exact calendar schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/forward_start_option/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    OptionSide Side,
    bool SeparateCallRounding = false
>
struct ForwardStartOptionPathPolicy {
    using ProductParameters = ForwardStartOptionParameters;
    using Calendar = simulation::StaticCalendar<2U>;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float moneyness;
        float discount;
    };

    struct Handler {
        float reset_spot = 0.0f;

        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            float spot
        ) {
            if (observation == 0U) reset_spot = spot;
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {{product.reset_time_days, product.maturity_days - product.reset_time_days}};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.moneyness,
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
        const typename StatePolicy::State& terminal,
        const Handler& handler
    ) {
        const float terminal_spot = StatePolicy::spot(terminal);
        if constexpr (Side == OptionSide::call) {
            const float reset_strike = SeparateCallRounding
                ? __fmul_rn(product.moneyness, handler.reset_spot)
                : product.moneyness * handler.reset_spot;
            return product.discount * fmaxf(
                terminal_spot - reset_strike,
                0.0f
            );
        } else {
            return product.discount * fmaxf(
                product.moneyness * handler.reset_spot - terminal_spot,
                0.0f
            );
        }
    }
};

namespace detail {

template<equity::SpotDynamicsPolicy Dynamics>
struct ForwardStartObservationHandler {
    float reset_spot = 0.0f;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        if (observation == 0U) {
            reset_spot = Dynamics::spot(state);
            return true;
        }
        return false;
    }
};

}  // namespace detail

template<
    simulation::TwoDateSchedulePolicy SchedulePolicy,
    OptionSide Side,
    bool SeparateCallRounding = false
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct ForwardStartOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ForwardStartOptionParameters;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float moneyness;
        float discount;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const typename Schedule::Calendar calendar{{
            product.reset_time_days,
            product.maturity_days - product.reset_time_days,
        }};
        return {
            Schedule::prepare(
                model,
                calendar,
                time_configuration
            ),
            product.moneyness,
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
        detail::ForwardStartObservationHandler<Dynamics> handler;
        const typename Dynamics::State terminal = Schedule::simulate(
            row.schedule,
            key,
            path,
            handler
        );
        const float terminal_spot = Dynamics::spot(terminal);
        if constexpr (Side == OptionSide::call) {
            if constexpr (SeparateCallRounding) {
                const float reset_strike = __fmul_rn(
                    row.moneyness,
                    handler.reset_spot
                );
                return row.discount * fmaxf(
                    __fsub_rn(terminal_spot, reset_strike),
                    0.0f
                );
            }
            return row.discount * fmaxf(
                terminal_spot - row.moneyness * handler.reset_spot,
                0.0f
            );
        } else {
            return row.discount * fmaxf(
                row.moneyness * handler.reset_spot - terminal_spot,
                0.0f
            );
        }
    }

};

}  // namespace ai_factory::workbench::product
