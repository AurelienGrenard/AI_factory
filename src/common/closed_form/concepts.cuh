// Compile-time contract for scalar closed-form pricing policies.
#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ai_factory::workbench::closed_form {

inline constexpr std::size_t kMaximumPreparedRowBytes = 256U;

template<typename Pricing>
concept ClosedFormPricingPolicy =
    std::is_trivially_copyable_v<typename Pricing::DeviceInputs>
    && std::is_trivially_copyable_v<typename Pricing::TimeConfiguration>
    && std::is_trivially_copyable_v<typename Pricing::PreparedRow>
    && sizeof(typename Pricing::PreparedRow) <= kMaximumPreparedRowBytes
    && requires(
        const typename Pricing::DeviceInputs& inputs,
        const typename Pricing::TimeConfiguration& time_configuration,
        const typename Pricing::PreparedRow& row,
        std::size_t result_index,
        std::size_t result_count
    ) {
        { inputs.validate(result_count) } -> std::same_as<void>;
        {
            inputs.template prepare_row<Pricing>(
                result_index,
                time_configuration
            )
        } -> std::same_as<typename Pricing::PreparedRow>;
        {
            validate_time_configuration(time_configuration)
        } -> std::same_as<void>;
        {
            Pricing::evaluate_price(row)
        } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::closed_form
