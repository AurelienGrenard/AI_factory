// Two-date forward-start payoff composed with an exact calendar schedule.
#pragma once

#include "common/device_inputs.cuh"

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "common/simulation/schedule.cuh"
#include "product/forward_start_option/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {
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
            product.reset_time,
            product.maturity - product.reset_time,
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
