// Host arithmetic shared by native launches and the offline planner.
#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {
inline constexpr unsigned int kDefaultSensitivityBatchSize = 1U;
inline constexpr unsigned int kDefaultThreadsPerBlock = 256U;
inline void validate_sensitivity_batch_size(unsigned int width) {
    if (width != 1U && width != 2U && width != 4U)
        throw std::invalid_argument("Sensitivity batch size must be 1, 2 or 4.");
}
inline std::size_t sensitivity_batch_count(std::size_t count, unsigned int width) {
    validate_sensitivity_batch_size(width);
    // Empty selection still owns one central price task per row.
    return count == 0U ? 1U : count / width + (count % width != 0U);
}
inline std::size_t gradient_task_count(std::size_t rows, std::size_t batches) {
    if (rows == 0U || batches == 0U) throw std::invalid_argument("Gradient work must be nonempty.");
    if (rows > std::numeric_limits<std::size_t>::max() / batches)
        throw std::overflow_error("Gradient row/batch task count overflow.");
    return rows*batches;
}
}  // namespace ai_factory::workbench::price_gradients
