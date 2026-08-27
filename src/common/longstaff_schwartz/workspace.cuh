// Host-side workspace layout and memory-aware batch planning for LSM pricers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz {

// Locate one typed array inside a contiguous device workspace.
struct WorkspaceRegion {
    std::size_t offset;
};

// Describe one model-specific state array stored at every exercise observation.
struct StateFieldDescriptor {
    std::size_t value_size;
    std::size_t alignment;
};

// Describe sizes that vary with the prepared row, state, and regression basis.
struct WorkspaceDescriptor {
    std::size_t prepared_row_size;
    std::size_t prepared_row_alignment;
    std::vector<StateFieldDescriptor> state_fields;
    std::size_t basis_size;
    std::size_t regression_value_count;
};

// All common arrays required by one proposed early-exercise batch.
struct WorkspaceLayout {
    WorkspaceRegion prepared_rows;
    WorkspaceRegion state_offsets;
    std::vector<WorkspaceRegion> state_fields;
    WorkspaceRegion cashflows;
    WorkspaceRegion regression_partials;
    WorkspaceRegion regression_coefficients;
    WorkspaceRegion regression_statuses;
    WorkspaceRegion regression_diagnostics;
    WorkspaceRegion moment_partials;
    std::size_t total_bytes;
};

// Schedule and state-storage requirement for one result row.
struct EarlyExerciseRowPlan {
    std::uint32_t regression_count;
    std::size_t state_value_count;
};

// One consecutive group of result rows that fits in the workspace.
struct BatchPlan {
    std::size_t result_offset;
    std::size_t result_count;
    std::size_t state_value_count;
    std::uint32_t maximum_regression_count;
};

// Complete memory-aware plan and maximum allocation dimensions for one launch.
struct ExecutionPlan {
    WorkspaceDescriptor descriptor;
    std::vector<EarlyExerciseRowPlan> rows;
    std::vector<BatchPlan> batches;
    std::size_t maximum_workspace_bytes;
    std::size_t maximum_prices_per_batch;
};

WorkspaceLayout make_workspace_layout(
    const WorkspaceDescriptor& descriptor,
    std::size_t batch_size,
    std::size_t state_value_count,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const char* product_name
);

ExecutionPlan plan_batches(
    const std::vector<EarlyExerciseRowPlan>& rows,
    const WorkspaceDescriptor& descriptor,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t workspace_budget,
    const char* product_name
);

template <typename Value>
Value* workspace_pointer(
    unsigned char* workspace,
    const WorkspaceRegion& region
) {
    return reinterpret_cast<Value*>(workspace + region.offset);
}

}  // namespace ai_factory::workbench::longstaff_schwartz
