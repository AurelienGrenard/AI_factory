// Selected-gradient additions to the common Longstaff-Schwartz workspace.
#pragma once

#include "common/check_cuda.cuh"

#include <algorithm>
#include <cstddef>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

inline std::size_t moment_value_count(std::size_t sensitivity_count) {
    return checked_workspace_product(
        2U,
        std::max<std::size_t>(sensitivity_count, 1U),
        "LSM gradient moment count exceeds size_t."
    );
}

inline std::size_t moment_value_count(
    std::size_t sensitivity_count,
    std::size_t moments_per_sensitivity
) {
    return checked_workspace_product(
        moments_per_sensitivity,
        std::max<std::size_t>(sensitivity_count, 1U),
        "LSM sensitivity moment count exceeds size_t."
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
