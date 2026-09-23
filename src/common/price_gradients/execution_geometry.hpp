// Work-count arithmetic shared by compact sensitivity kernels and planners.
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {

inline constexpr unsigned int kSensitivitiesPerBlock = 1U;
inline constexpr unsigned int kDefaultThreadsPerBlock = 256U;

inline std::size_t sensitivity_task_count(
    std::size_t rows,
    std::size_t sensitivities
) {
    if (rows == 0U) {
        throw std::invalid_argument("Sensitivity work must contain a row.");
    }
    const auto tasks_per_row = sensitivities == 0U ? 1U : sensitivities;
    if (rows > std::numeric_limits<std::size_t>::max() / tasks_per_row) {
        throw std::overflow_error("Sensitivity task count overflow.");
    }
    return rows * tasks_per_row;
}

}  // namespace ai_factory::workbench::price_gradients
