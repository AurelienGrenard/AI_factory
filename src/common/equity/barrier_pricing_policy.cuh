// Compile-time pricing policies shared by identical discrete barrier payoffs.
#pragma once

#include "common/equity/barrier_observation_handlers.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/option_payoff.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

template<
    DenseEquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    typename ProductParametersT,
    OptionSide Side,
    BarrierDirection Direction,
    bool KnockIn
>
requires DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct SingleBarrierOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float strike;
        float barrier;
        float discount;
    };

    static_assert(std::is_trivially_copyable_v<ProductParameters>);
    static_assert(std::is_trivially_copyable_v<DeviceInputs>);
    static_assert(std::is_trivially_copyable_v<PreparedRow>);

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::uint64_t seed
    ) {
        const typename Schedule::Definition definition{product.maturity};
        return {
            Schedule::prepare(model, definition, configuration, inputs.schedule),
            philox::make_key(seed),
            product.strike,
            product.barrier,
            Discount::discount_factor(
                model,
                inputs.discount,
                Schedule::total_year_fraction(definition, configuration)
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        if constexpr (KnockIn) {
            KnockInBarrierObservationHandler<
                Dynamics,
                Direction
            > handler{row.barrier};
            const typename Dynamics::State terminal = Schedule::simulate(
                row.schedule,
                row.key,
                path,
                handler
            );
            if (!handler.activated) return 0.0f;
            const float terminal_spot = Dynamics::spot(terminal);
            return row.discount
                * option_payoff<Side>(terminal_spot, row.strike);
        } else {
            KnockOutBarrierObservationHandler<
                Dynamics,
                Direction
            > handler{row.barrier};
            Schedule::simulate(row.schedule, row.key, path, handler);
            if (barrier_breached<Direction>(
                    handler.terminal_spot,
                    row.barrier
                )) {
                return 0.0f;
            }
            return row.discount
                * option_payoff<Side>(handler.terminal_spot, row.strike);
        }
    }

    static void validate_configuration(
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::size_t monte_carlo_paths_per_price
    ) {
        Schedule::validate_configuration(
            configuration,
            inputs.schedule,
            monte_carlo_paths_per_price
        );
        Discount::validate_inputs(inputs.discount);
    }
};

template<
    DenseEquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    typename ProductParametersT,
    bool PaysOnHit
>
requires DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct UpTouchPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = ProductParametersT;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float barrier;
        float discounted_cash_payoff;
    };

    static_assert(std::is_trivially_copyable_v<ProductParameters>);
    static_assert(std::is_trivially_copyable_v<DeviceInputs>);
    static_assert(std::is_trivially_copyable_v<PreparedRow>);

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::uint64_t seed
    ) {
        const typename Schedule::Definition definition{product.maturity};
        return {
            Schedule::prepare(model, definition, configuration, inputs.schedule),
            philox::make_key(seed),
            product.barrier,
            product.cash_payoff * Discount::discount_factor(
                model,
                inputs.discount,
                Schedule::total_year_fraction(definition, configuration)
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        KnockOutBarrierObservationHandler<
            Dynamics,
            BarrierDirection::up
        > handler{row.barrier};
        Schedule::simulate(row.schedule, row.key, path, handler);
        const bool hit = barrier_breached<BarrierDirection::up>(
            handler.terminal_spot,
            row.barrier
        );
        return hit == PaysOnHit ? row.discounted_cash_payoff : 0.0f;
    }

    static void validate_configuration(
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::size_t monte_carlo_paths_per_price
    ) {
        Schedule::validate_configuration(
            configuration,
            inputs.schedule,
            monte_carlo_paths_per_price
        );
        Discount::validate_inputs(inputs.discount);
    }
};

}  // namespace ai_factory::workbench::equity
