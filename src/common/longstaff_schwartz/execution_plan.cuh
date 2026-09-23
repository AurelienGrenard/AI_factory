// Pricing-policy adaptation over the common memory-aware batch planner.
#pragma once

#include "common/longstaff_schwartz/concepts.cuh"
#include "common/longstaff_schwartz/workspace.cuh"

#include <cstddef>
#include <limits>
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
    const typename PricingPolicy::Schedule::TimeConfiguration&
        time_configuration,
    const char* product_name,
    std::size_t maximum_batch_size =
        std::numeric_limits<std::size_t>::max()
) {
    inputs.validate(result_count);

    WorkspaceDescriptor descriptor{
        sizeof(typename PricingPolicy::PreparedRow),
        alignof(typename PricingPolicy::PreparedRow),
        PricingPolicy::state_field_descriptors(),
        Regressor::kBasisSize,
        Regressor::kRegressionValueCount,
    };
    if constexpr (requires { PricingPolicy::moment_value_count(inputs); }) {
        descriptor.moment_value_count =
            PricingPolicy::moment_value_count(inputs);
    }
    if constexpr (requires { PricingPolicy::observation_field_descriptors(); }) {
        descriptor.observation_fields = PricingPolicy::observation_field_descriptors();
    }
    if constexpr (requires { PricingPolicy::path_field_descriptors(); }) {
        descriptor.path_fields = PricingPolicy::path_field_descriptors();
    }
    if constexpr (requires { PricingPolicy::row_field_descriptors(); }) {
        descriptor.row_fields = PricingPolicy::row_field_descriptors();
    }

    std::vector<EarlyExerciseRowPlan> rows;
    rows.reserve(result_count);
    for (std::size_t result_index = 0U;
         result_index < result_count;
         ++result_index) {
        rows.push_back(PricingPolicy::plan_row(
            inputs,
            result_index,
            paths_per_price,
            time_configuration
        ));
    }

    return plan_batches(
        rows,
        descriptor,
        paths_per_price,
        blocks_per_price,
        workspace_budget,
        product_name,
        maximum_batch_size
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz
