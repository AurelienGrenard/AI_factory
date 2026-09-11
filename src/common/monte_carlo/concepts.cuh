// Compile-time contract for scalar Monte Carlo pricing policies.
#pragma once

#include "common/device_inputs.cuh"
#include "common/simulation/concepts.cuh"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo {

// One prepared price row is stored once per block in static shared memory.
inline constexpr std::size_t kMaximumSharedPreparedRowBytes = 2048U;

// One pricing policy binds a product to a simulation schedule and exposes the
// minimal interface consumed by the generic one-block-per-price kernel.
template<typename PricingPolicy>
concept MonteCarloLaunchPolicy =
    simulation::SchedulePolicy<typename PricingPolicy::Schedule>
    && std::is_trivially_copyable_v<typename PricingPolicy::DeviceInputs>
    && std::is_trivially_copyable_v<
        typename PricingPolicy::ProductParameters
    >
    && std::is_trivially_copyable_v<typename PricingPolicy::PreparedRow>
    && requires(
        const typename PricingPolicy::DeviceInputs& inputs,
        const typename PricingPolicy::HostInputs& host_inputs,
        const typename PricingPolicy::Schedule::TimeConfiguration&
            time_configuration,
        std::size_t result_count
    ) {
        { inputs.validate(result_count) } -> std::same_as<void>;
        {
            host_inputs.validate(0U, time_configuration)
        } -> std::same_as<void>;
        {
            inputs.template prepare_row<PricingPolicy>(
                0U,
                time_configuration
            )
        } -> std::same_as<typename PricingPolicy::PreparedRow>;
    };

template<typename PricingPolicy>
concept ScalarMonteCarloPricingPolicy = MonteCarloLaunchPolicy<PricingPolicy>
    && requires(const typename PricingPolicy::PreparedRow& row,
                philox::PhiloxKey key, std::size_t path) {
        { PricingPolicy::evaluate_path(row, key, path) } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::monte_carlo
