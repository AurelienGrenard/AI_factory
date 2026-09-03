// Compile-time pricing policies shared by identical discrete barrier payoffs.
#pragma once

#include "common/device_inputs.cuh"

#include "common/simulation/barrier_handlers.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/equity/observables.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/barrier.cuh"
#include "common/payoff/vanilla_option.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

// Model-independent scalar-observation policy for block-level path engines.
// The engine adapts its state to spot values before invoking this handler.
template<
    typename ProductParametersT,
    OptionSide Side,
    payoff::BarrierDirection Direction,
    bool KnockIn
>
struct SingleBarrierOptionPathPolicy {
    using ProductParameters = ProductParametersT;
    using Calendar = simulation::MaturityCalendar;
    static constexpr ObservationCoordinate kObservationCoordinate =
        ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float barrier;
        float discount;
    };

    struct Handler {
        float barrier;
        float terminal_value = 0.0f;
        bool activated = false;

        __device__ __forceinline__ bool observe(float value) {
            terminal_value = value;
            const bool breached = payoff::barrier_breached<Direction>(
                value,
                barrier
            );
            if constexpr (KnockIn) {
                activated = activated || breached;
                return true;
            } else {
                activated = breached;
                return !breached;
            }
        }

        __device__ __forceinline__ bool on_initial_value(float value) {
            return observe(value);
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float value
        ) {
            return observe(value);
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
        ProductPreparationContext context
    ) {
        return {
            product.strike,
            product.barrier,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {product.barrier};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        if constexpr (KnockIn) {
            if (!handler.activated) return 0.0f;
        } else {
            if (handler.activated) return 0.0f;
        }
        return product.discount * payoff::vanilla_option_payoff<Side>(
            handler.terminal_value,
            product.strike
        );
    }
};

static_assert(std::is_trivially_copyable_v<
    SingleBarrierOptionPathPolicy<
        int,
        OptionSide::call,
        payoff::BarrierDirection::up,
        false
    >::Handler
>);

template<typename ProductParametersT, bool PaysOnHit>
struct UpTouchPathPolicy {
    using ProductParameters = ProductParametersT;
    using Calendar = simulation::MaturityCalendar;
    static constexpr ObservationCoordinate kObservationCoordinate =
        ObservationCoordinate::spot;

    struct PreparedProduct {
        float barrier;
        float discounted_cash_payoff;
    };

    struct Handler {
        float barrier;
        bool activated = false;

        __device__ __forceinline__ bool observe(float spot) {
            activated = activated || spot >= barrier;
            return !activated;
        }

        __device__ __forceinline__ bool on_initial_value(float spot) {
            return observe(spot);
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
        ProductPreparationContext context
    ) {
        return {
            product.barrier,
            product.cash_payoff * expf(
                -model.risk_free_rate * context.maturity_years
            ),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {product.barrier};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        const bool pays = PaysOnHit ? handler.activated : !handler.activated;
        return pays ? product.discounted_cash_payoff : 0.0f;
    }
};

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
        const typename Schedule::Calendar calendar{product.maturity_days};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.strike,
            product.barrier,
            constant_rate_discount_factor(
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
        const typename Schedule::Calendar calendar{product.maturity_days};
        return {
            Schedule::prepare(model, calendar, time_configuration),
            product.barrier,
            product.cash_payoff * constant_rate_discount_factor(
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
