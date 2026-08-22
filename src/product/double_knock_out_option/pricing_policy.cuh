// Double-knock-out payoff composed with a dense equity schedule.
#pragma once

#include "common/equity/barrier_observation_handlers.cuh"
#include "common/equity/discount.cuh"
#include "common/equity/option_payoff.cuh"
#include "product/double_knock_out_option/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

template<
    equity::DenseEquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    OptionSide Side
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct DoubleKnockOutOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = DoubleKnockOutOptionParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float strike;
        float lower_barrier;
        float upper_barrier;
        float discount;
    };

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
            product.lower_barrier,
            product.upper_barrier,
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
        equity::DoubleKnockOutObservationHandler<Dynamics> handler{
            row.lower_barrier,
            row.upper_barrier,
        };
        Schedule::simulate(row.schedule, row.key, path, handler);
        if (handler.terminal_spot <= row.lower_barrier
            || handler.terminal_spot >= row.upper_barrier) {
            return 0.0f;
        }
        return row.discount * equity::option_payoff<Side>(
            handler.terminal_spot,
            row.strike
        );
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

}  // namespace ai_factory::workbench::product
