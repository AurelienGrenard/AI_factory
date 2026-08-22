// Cash-or-nothing payoff composed with a terminal equity schedule.
#pragma once

#include "common/equity/discount.cuh"
#include "common/equity/observation_handlers.cuh"
#include "common/option_side.cuh"
#include "product/digital_option/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

template<
    equity::TerminalEquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy,
    OptionSide Side
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct DigitalOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = DigitalOptionParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float strike;
        float discounted_cash_payoff;
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
        const typename Dynamics::State terminal = Schedule::simulate_terminal(
            row.schedule,
            row.key,
            path
        );
        const float terminal_spot = Dynamics::spot(terminal);
        const bool pays = Side == OptionSide::call
            ? terminal_spot > row.strike
            : terminal_spot < row.strike;
        return pays ? row.discounted_cash_payoff : 0.0f;
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
