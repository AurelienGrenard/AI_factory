// Host-side aligned workspace construction and consecutive batch planning.
#include "common/longstaff_schwartz/workspace.cuh"

#include "common/check_cuda.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz {
namespace {

constexpr std::size_t kMomentValueCount = 2U;

std::string workspace_error(
    const char* product_name,
    const char* region_name
) {
    return std::string(product_name) + ' ' + region_name
        + " exceed size_t.";
}

std::size_t aligned_offset(
    std::size_t offset,
    std::size_t alignment,
    const char* product_name
) {
    if (alignment == 0U) {
        throw std::invalid_argument("Workspace alignment must be positive.");
    }
    const std::size_t remainder = offset % alignment;
    if (remainder == 0U) return offset;
    const std::size_t padding = alignment - remainder;
    if (offset > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::overflow_error(
            std::string(product_name) + " workspace offset exceeds size_t."
        );
    }
    return offset + padding;
}

WorkspaceRegion append_region(
    std::size_t& cursor,
    std::size_t value_count,
    std::size_t value_size,
    std::size_t alignment,
    const char* product_name,
    const char* region_name
) {
    cursor = aligned_offset(cursor, alignment, product_name);
    const std::size_t offset = cursor;
    const std::string error = workspace_error(product_name, region_name);
    const std::size_t bytes = checked_workspace_product(
        value_count, value_size, error.c_str()
    );
    if (cursor > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error(error);
    }
    cursor += bytes;
    return {offset};
}

void validate_descriptor(const WorkspaceDescriptor& descriptor) {
    if (descriptor.prepared_row_size == 0U
        || descriptor.prepared_row_alignment == 0U
        || descriptor.state_fields.empty()
        || descriptor.basis_size == 0U
        || descriptor.regression_value_count == 0U) {
        throw std::invalid_argument(
            "An early-exercise workspace descriptor contains an empty dimension."
        );
    }
    for (const StateFieldDescriptor& field : descriptor.state_fields) {
        if (field.value_size == 0U || field.alignment == 0U) {
            throw std::invalid_argument(
                "An early-exercise state field contains an empty dimension."
            );
        }
    }
}

}  // namespace

WorkspaceLayout make_workspace_layout(
    const WorkspaceDescriptor& descriptor,
    std::size_t batch_size,
    std::size_t state_value_count,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const char* product_name
) {
    validate_descriptor(descriptor);
    const std::size_t path_value_count = checked_workspace_product(
        batch_size,
        paths_per_price,
        workspace_error(product_name, "cashflow count").c_str()
    );
    const std::size_t partial_block_count = checked_workspace_product(
        batch_size,
        blocks_per_price,
        workspace_error(product_name, "partial block count").c_str()
    );
    const std::size_t regression_partial_count = checked_workspace_product(
        partial_block_count,
        descriptor.regression_value_count,
        workspace_error(product_name, "regression partial count").c_str()
    );
    const std::size_t coefficient_count = checked_workspace_product(
        batch_size,
        descriptor.basis_size,
        workspace_error(product_name, "coefficient count").c_str()
    );
    const std::size_t moment_partial_count = checked_workspace_product(
        partial_block_count,
        kMomentValueCount,
        workspace_error(product_name, "moment partial count").c_str()
    );

    std::size_t cursor = 0U;
    WorkspaceLayout layout{};
    layout.prepared_rows = append_region(
        cursor,
        batch_size,
        descriptor.prepared_row_size,
        descriptor.prepared_row_alignment,
        product_name,
        "prepared rows"
    );
    layout.exercise_counts = append_region(
        cursor,
        batch_size,
        sizeof(std::uint32_t),
        alignof(std::uint32_t),
        product_name,
        "exercise counts"
    );
    layout.state_offsets = append_region(
        cursor,
        batch_size,
        sizeof(std::size_t),
        alignof(std::size_t),
        product_name,
        "state offsets"
    );
    layout.state_fields.reserve(descriptor.state_fields.size());
    for (const StateFieldDescriptor& field : descriptor.state_fields) {
        layout.state_fields.push_back(append_region(
            cursor,
            state_value_count,
            field.value_size,
            field.alignment,
            product_name,
            "state values"
        ));
    }
    layout.cashflows = append_region(
        cursor,
        path_value_count,
        sizeof(float),
        alignof(float),
        product_name,
        "cashflows"
    );
    layout.regression_partials = append_region(
        cursor,
        regression_partial_count,
        sizeof(double),
        alignof(double),
        product_name,
        "regression partials"
    );
    layout.regression_coefficients = append_region(
        cursor,
        coefficient_count,
        sizeof(double),
        alignof(double),
        product_name,
        "regression coefficients"
    );
    layout.regression_valid = append_region(
        cursor,
        batch_size,
        sizeof(std::uint32_t),
        alignof(std::uint32_t),
        product_name,
        "regression flags"
    );
    layout.moment_partials = append_region(
        cursor,
        moment_partial_count,
        sizeof(double),
        alignof(double),
        product_name,
        "moment partials"
    );
    layout.total_bytes = cursor;
    return layout;
}

ExecutionPlan plan_batches(
    const std::vector<EarlyExerciseRowPlan>& rows,
    const WorkspaceDescriptor& descriptor,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t workspace_budget,
    const char* product_name
) {
    if (rows.empty()) {
        throw std::invalid_argument("Early-exercise batch planning requires rows.");
    }
    ExecutionPlan plan{};
    std::size_t result_offset = 0U;

    while (result_offset < rows.size()) {
        std::size_t batch_size = 0U;
        std::size_t state_value_count = 0U;
        std::uint32_t maximum_exercise_count = 0U;

        while (result_offset + batch_size < rows.size()) {
            const EarlyExerciseRowPlan& row = rows[result_offset + batch_size];
            if (state_value_count
                > std::numeric_limits<std::size_t>::max()
                    - row.state_value_count) {
                throw std::overflow_error(
                    std::string(product_name)
                    + " batch state count exceeds size_t."
                );
            }
            const std::size_t candidate_state_values =
                state_value_count + row.state_value_count;
            const WorkspaceLayout candidate = make_workspace_layout(
                descriptor,
                batch_size + 1U,
                candidate_state_values,
                paths_per_price,
                blocks_per_price,
                product_name
            );
            if (candidate.total_bytes > workspace_budget) break;

            ++batch_size;
            state_value_count = candidate_state_values;
            maximum_exercise_count = std::max(
                maximum_exercise_count, row.exercise_count
            );
        }

        if (batch_size == 0U) {
            const WorkspaceLayout required = make_workspace_layout(
                descriptor,
                1U,
                rows[result_offset].state_value_count,
                paths_per_price,
                blocks_per_price,
                product_name
            );
            throw std::runtime_error(
                std::string(product_name) + " result row "
                + std::to_string(result_offset) + " requires "
                + std::to_string(required.total_bytes)
                + " workspace bytes, but only "
                + std::to_string(workspace_budget)
                + " bytes are available. Reduce paths per price."
            );
        }

        const WorkspaceLayout layout = make_workspace_layout(
            descriptor,
            batch_size,
            state_value_count,
            paths_per_price,
            blocks_per_price,
            product_name
        );
        plan.batches.push_back({
            result_offset,
            batch_size,
            state_value_count,
            maximum_exercise_count,
        });
        plan.maximum_workspace_bytes = std::max(
            plan.maximum_workspace_bytes, layout.total_bytes
        );
        plan.maximum_prices_per_batch = std::max(
            plan.maximum_prices_per_batch, batch_size
        );
        result_offset += batch_size;
    }
    return plan;
}

}  // namespace ai_factory::workbench::longstaff_schwartz
