// Compile-time pricing policies shared by identical discrete barrier payoffs.
#pragma once

#include "common/device_inputs.cuh"

#include "common/simulation/barrier_handlers.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/observables.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/barrier.cuh"
#include "common/payoff/vanilla_option.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::equity {

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    typename ProductParametersT,
    OptionSide Side,
    payoff::BarrierDirection Direction,
    bool KnockIn
>
requires SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct SingleBarrierOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float strike;
        float barrier;
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
            product.barrier,
            constant_rate_discount_factor(
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
        if constexpr (KnockIn) {
            simulation::KnockInBarrierObservationHandler<
                Dynamics,
                SpotObservable<Dynamics>,
                Direction
            > handler{row.barrier};
            const typename Dynamics::State terminal = Schedule::simulate(
                row.schedule,
                key,
                path,
                handler
            );
            if (!handler.activated) return 0.0f;
            const float terminal_spot = Dynamics::spot(terminal);
            return row.discount
                * payoff::vanilla_option_payoff<Side>(
                    terminal_spot,
                    row.strike
                );
        } else {
            simulation::KnockOutBarrierObservationHandler<
                Dynamics,
                SpotObservable<Dynamics>,
                Direction
            > handler{row.barrier};
            Schedule::simulate(row.schedule, key, path, handler);
            if (payoff::barrier_breached<Direction>(
                    handler.terminal_value,
                    row.barrier
                )) {
                return 0.0f;
            }
            return row.discount
                * payoff::vanilla_option_payoff<Side>(
                    handler.terminal_value,
                    row.strike
                );
        }
    }

};

template<
    simulation::DenseSchedulePolicy SchedulePolicy,
    typename ProductParametersT,
    bool PaysOnHit
>
requires SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
struct UpTouchPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using DeviceInputs =
        ModelProductDeviceInputs<ModelParameters, ProductParameters>;
    using TimeConfiguration = typename Schedule::TimeConfiguration;

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        float barrier;
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
            product.barrier,
            product.cash_payoff * constant_rate_discount_factor(
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
        simulation::KnockOutBarrierObservationHandler<
            Dynamics,
            SpotObservable<Dynamics>,
            payoff::BarrierDirection::up
        > handler{row.barrier};
        Schedule::simulate(row.schedule, key, path, handler);
        const bool hit = payoff::barrier_breached<
            payoff::BarrierDirection::up
        >(
            handler.terminal_value,
            row.barrier
        );
        return hit == PaysOnHit ? row.discounted_cash_payoff : 0.0f;
    }

};

}  // namespace ai_factory::workbench::equity
