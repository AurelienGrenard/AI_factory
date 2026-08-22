// Athena payoff composed with arbitrary equity schedule and discount policies.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/discount.cuh"
#include "product/athena_autocall/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {
namespace detail {

template<equity::EquitySchedulePolicy Schedule>
struct AthenaAutocallObservationHandler {
    using Dynamics = typename Schedule::Dynamics;

    std::uint32_t observation_count;
    float autocall_barrier;
    float protection_barrier;
    float gain_per_observation;
    float discount_per_observation;
    float discount = 1.0f;
    float accumulated_gain = 0.0f;
    float discounted_payoff = 0.0f;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        discount *= discount_per_observation;
        accumulated_gain += gain_per_observation;
        const float current_spot = Dynamics::spot(state);
        const bool maturity = observation + 1U == observation_count;

        if (!maturity && current_spot >= autocall_barrier) {
            discounted_payoff = discount * (1.0f + accumulated_gain);
            return false;
        }
        if (maturity) {
            const float redemption =
                current_spot >= autocall_barrier
                ? 1.0f + accumulated_gain
                : (
                    current_spot >= protection_barrier
                    ? 1.0f
                    : current_spot
                );
            discounted_payoff = discount * redemption;
            return false;
        }
        return true;
    }
};

}  // namespace detail

template<
    equity::EquitySchedulePolicy SchedulePolicy,
    typename DiscountPolicy
>
requires equity::DiscountPolicyFor<
    DiscountPolicy,
    typename SchedulePolicy::Dynamics
>
struct AthenaAutocallPricingPolicy {
    using Schedule = SchedulePolicy;
    using Discount = DiscountPolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = AthenaAutocallParameters;
    using PricingConfiguration = typename Schedule::Configuration;

    struct DeviceInputs {
        typename Schedule::DeviceInputs schedule;
        typename Discount::DeviceInputs discount;
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        float autocall_barrier;
        float protection_barrier;
        float gain_per_observation;
        float discount_per_observation;
    };

    static_assert(std::is_trivially_copyable_v<DeviceInputs>);
    static_assert(std::is_trivially_copyable_v<PreparedRow>);
    static_assert(
        sizeof(PreparedRow) <= equity::kMaximumPreparedRowBytes
    );

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const PricingConfiguration& configuration,
        const DeviceInputs& inputs,
        std::uint64_t seed
    ) {
        const typename Schedule::Definition definition{
            product.observation_interval,
            product.maturity / product.observation_interval,
        };
        const float observation_years =
            Schedule::interval_year_fraction(
                definition,
                0U,
                configuration
            );
        return {
            Schedule::prepare(
                model,
                definition,
                configuration,
                inputs.schedule
            ),
            philox::make_key(seed),
            product.autocall_barrier,
            product.protection_barrier,
            product.annual_coupon_rate * observation_years,
            Discount::discount_factor(
                model,
                inputs.discount,
                observation_years
            ),
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        std::size_t path
    ) {
        detail::AthenaAutocallObservationHandler<Schedule> handler{
            Schedule::observation_count(row.schedule),
            row.autocall_barrier,
            row.protection_barrier,
            row.gain_per_observation,
            row.discount_per_observation,
        };
        Schedule::simulate(
            row.schedule,
            row.key,
            path,
            handler
        );
        return handler.discounted_payoff;
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
