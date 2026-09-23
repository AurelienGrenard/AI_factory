// Adapt an existing fixed-income closed-form policy to prepared gradient rows.
#pragma once

#include "common/fixed_income/price_gradients/device_preparation.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::fixed_income::price_gradients {

template<
    typename PricingPolicy,
    typename Model,
    typename Product,
    typename Context
>
struct ScenarioClosedFormPolicy {
    using InputRow = Scenario<Model, Product>;
    using PreparedRow = typename PricingPolicy::PreparedRow;

    __device__ __forceinline__ static PreparedRow prepare(
        const InputRow& input
    ) {
        return PricingPolicy::prepare_row(
            input.model,
            input.product,
            Context{},
            typename PricingPolicy::TimeConfiguration{input.day_fraction}
        );
    }

    __device__ __forceinline__ static float evaluate(
        const InputRow& input
    ) {
        return evaluate(prepare(input));
    }

    __device__ __forceinline__ static float evaluate(
        const PreparedRow& row
    ) {
        return PricingPolicy::evaluate_price(row);
    }

    inline static std::size_t required_shared_memory_bytes(
        std::uint32_t workspace_capacity
    ) {
        return PricingPolicy::required_shared_memory_bytes(
            workspace_capacity
        );
    }

    __device__ __forceinline__ static float evaluate(
        const PreparedRow& row,
        std::byte* workspace,
        std::uint32_t workspace_capacity
    ) {
        return PricingPolicy::evaluate_price(
            row, workspace, workspace_capacity
        );
    }
};

}  // namespace ai_factory::workbench::fixed_income::price_gradients
