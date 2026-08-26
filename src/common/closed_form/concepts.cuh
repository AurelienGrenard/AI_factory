// Compile-time contract for scalar closed-form pricing policies.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::closed_form {

// Closed-form rows are local to one pricing thread.
inline constexpr std::size_t kMaximumThreadPreparedRowBytes = 256U;

// Cooperative closed-form rows are stored once per CUDA block.
inline constexpr std::size_t kMaximumSharedPreparedRowBytes = 256U;

template<typename PricingPolicy>
concept ClosedFormPreparedRowPolicy =
    std::is_trivially_copyable_v<typename PricingPolicy::DeviceInputs>
    && std::is_trivially_copyable_v<
        typename PricingPolicy::TimeConfiguration
    >
    && std::is_trivially_copyable_v<typename PricingPolicy::PreparedRow>
    && requires(
        const typename PricingPolicy::DeviceInputs& inputs,
        const typename PricingPolicy::TimeConfiguration& time_configuration,
        std::size_t result_index,
        std::size_t result_count
    ) {
        { inputs.validate(result_count) } -> std::same_as<void>;
        {
            inputs.template prepare_row<PricingPolicy>(
                result_index,
                time_configuration
            )
        } -> std::same_as<typename PricingPolicy::PreparedRow>;
        {
            validate_time_configuration(time_configuration)
        } -> std::same_as<void>;
    };

template<typename PricingPolicy>
concept ClosedFormPricingPolicy =
    ClosedFormPreparedRowPolicy<PricingPolicy>
    && requires(
        const typename PricingPolicy::PreparedRow& row
    ) {
        {
            PricingPolicy::evaluate_price(row)
        } -> std::same_as<float>;
    };

template<typename PricingPolicy>
concept CooperativeClosedFormPricingPolicy =
    ClosedFormPreparedRowPolicy<PricingPolicy>
    && requires(
        const typename PricingPolicy::PreparedRow& row,
        std::byte* workspace,
        std::uint32_t workspace_capacity
    ) {
        {
            PricingPolicy::required_shared_memory_bytes(workspace_capacity)
        } -> std::same_as<std::size_t>;
        {
            PricingPolicy::evaluate_price(
                row,
                workspace,
                workspace_capacity
            )
        } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::closed_form
