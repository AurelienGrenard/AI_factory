// Adapt an existing fixed-income closed-form policy to prepared gradient rows.
#pragma once

#include "common/fixed_income/price_gradients/device_preparation.cuh"
#include "common/fixed_income/price_gradients/terminal_maturity.cuh"

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
    using BasePreparedRow = typename PricingPolicy::PreparedRow;
    using PreparedRow = TerminalMaturityPreparedRow<BasePreparedRow>;

    __device__ __forceinline__ static PreparedRow prepare(
        const InputRow& input
    ) {
        return apply_terminal_maturity(
            PricingPolicy::prepare_row(
                input.model,
                input.product,
                Context{},
                typename PricingPolicy::TimeConfiguration{
                    input.day_fraction
                }
            ),
            input.maturity_years,
            input.step_count != input.central_step_count
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


template<typename PricingPolicy, typename Model, typename Product>
struct ScalarScenarioClosedFormPolicy {
    using InputRow = Scenario<Model, Product>;
    using BasePreparedRow = typename PricingPolicy::PreparedRow;
    using PreparedRow = TerminalMaturityPreparedRow<BasePreparedRow>;

    __device__ __forceinline__ static PreparedRow prepare(
        const InputRow& input
    ) {
        return apply_terminal_maturity(
            PricingPolicy::prepare_row(
                input.model,
                input.product,
                typename PricingPolicy::TimeConfiguration{
                    input.day_fraction
                }
            ),
            input.maturity_years,
            input.step_count != input.central_step_count
        );
    }

    __device__ __forceinline__ static float evaluate(
        const InputRow& input
    ) {
        return PricingPolicy::evaluate_price(prepare(input));
    }
};

template<
    typename PricingPolicy,
    typename Model,
    typename Curve,
    typename Product
>
struct CurveScalarScenarioClosedFormPolicy {
    using InputRow = CurveScenario<Model, Curve, Product>;
    using BasePreparedRow = typename PricingPolicy::PreparedRow;
    using PreparedRow = TerminalMaturityPreparedRow<BasePreparedRow>;

    __device__ __forceinline__ static PreparedRow prepare(
        const InputRow& input
    ) {
        return apply_terminal_maturity(
            PricingPolicy::prepare_row(
                input.model,
                input.curve,
                input.product,
                typename PricingPolicy::TimeConfiguration{
                    input.day_fraction
                }
            ),
            input.maturity_years,
            input.step_count != input.central_step_count
        );
    }

    __device__ __forceinline__ static float evaluate(
        const InputRow& input
    ) {
        return PricingPolicy::evaluate_price(prepare(input));
    }
};

template<
    typename PricingPolicy,
    typename Model,
    typename Curve,
    typename Product,
    typename Context
>
struct CurveScenarioClosedFormPolicy {
    using InputRow = CurveScenario<Model, Curve, Product>;
    using BasePreparedRow = typename PricingPolicy::PreparedRow;
    using PreparedRow = TerminalMaturityPreparedRow<BasePreparedRow>;

    __device__ __forceinline__ static PreparedRow prepare(
        const InputRow& input
    ) {
        return apply_terminal_maturity(
            PricingPolicy::prepare_row(
                input.model,
                input.curve,
                input.product,
                Context{},
                typename PricingPolicy::TimeConfiguration{
                    input.day_fraction
                }
            ),
            input.maturity_years,
            input.step_count != input.central_step_count
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
