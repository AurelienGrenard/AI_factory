// Memory-planning helpers specific to selected frozen-exercise gradients.
#pragma once

#include "common/longstaff_schwartz/price_gradients/workspace.cuh"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

inline void validate_task_grid(
    std::size_t batch_size,
    std::size_t sensitivity_count,
    std::size_t maximum_grid_y
) {
    if (sensitivity_count == 0U) return;
    if (batch_size > maximum_grid_y / sensitivity_count) {
        throw std::overflow_error(
            "LSM gradient task count exceeds the current gridDim.y limit."
        );
    }
}

inline std::size_t maximum_batch_size(
    std::size_t sensitivity_count,
    std::size_t maximum_grid_y
) {
    if (maximum_grid_y == 0U) {
        throw std::invalid_argument("LSM gridDim.y limit must be positive.");
    }
    return sensitivity_count == 0U
        ? maximum_grid_y
        : maximum_grid_y / sensitivity_count;
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
