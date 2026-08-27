// Pricing-policy adaptation over the common memory-aware batch planner.
#pragma once

#include "common/longstaff_schwartz/concepts.cuh"
#include "common/longstaff_schwartz/workspace.cuh"

#include <cstddef>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz {

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
ExecutionPlan make_execution_plan(
    const typename PricingPolicy::HostInputs& inputs,
    std::size_t result_count,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t workspace_budget,
    const char* product_name
) {
    inputs.validate(result_count);

    WorkspaceDescriptor descriptor{
        sizeof(typename PricingPolicy::PreparedRow),
        alignof(typename PricingPolicy::PreparedRow),
        PricingPolicy::state_field_descriptors(),
        Regressor::kBasisSize,
        Regressor::kRegressionValueCount,
    };

    std::vector<EarlyExerciseRowPlan> rows;
    rows.reserve(result_count);
    for (std::size_t result_index = 0U;
         result_index < result_count;
         ++result_index) {
        rows.push_back(PricingPolicy::plan_row(
            inputs,
            result_index,
            paths_per_price
        ));
    }

    return plan_batches(
        rows,
        descriptor,
        paths_per_price,
        blocks_per_price,
        workspace_budget,
        product_name
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz
