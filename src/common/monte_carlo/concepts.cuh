// Compile-time contract for scalar Monte Carlo pricing policies.
#pragma once

#include "common/device_inputs.cuh"
#include "common/simulation/concepts.cuh"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo {

// One pricing policy binds a product to a simulation schedule and exposes the
// minimal interface consumed by the generic one-block-per-price kernel.
template<typename Pricing>
concept ScalarMonteCarloPricingPolicy =
    simulation::SchedulePolicy<typename Pricing::Schedule>
    && std::is_trivially_copyable_v<typename Pricing::DeviceInputs>
    && std::is_trivially_copyable_v<typename Pricing::ProductParameters>
    && std::is_trivially_copyable_v<typename Pricing::PreparedRow>
    && sizeof(typename Pricing::PreparedRow)
        <= simulation::kMaximumPreparedRowBytes
    && requires(
        const typename Pricing::DeviceInputs& inputs,
        const typename Pricing::Schedule::TimeConfiguration&
            time_configuration,
        const typename Pricing::PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        {
            inputs.template prepare_row<Pricing>(
                0U,
                time_configuration
            )
        } -> std::same_as<typename Pricing::PreparedRow>;
        {
            Pricing::evaluate_path(row, key, path)
        } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::monte_carlo
