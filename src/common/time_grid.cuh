// Shared integer-index time grids such as 1/252, 1/360, or 1/365.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::time_grid {

// Store the reciprocal once on the host so hot device code only multiplies.
struct TimeGrid {
    std::uint32_t steps_per_year;
    float step_size;

    __host__ __device__ explicit constexpr TimeGrid(
        std::uint32_t requested_steps_per_year
    ) noexcept
        : steps_per_year(requested_steps_per_year),
          step_size(
              requested_steps_per_year == 0U
                  ? 0.0f
                  : 1.0f / static_cast<float>(requested_steps_per_year)
          ) {}
};

__host__ __device__ constexpr float year_fraction(
    std::uint32_t index,
    TimeGrid grid
) noexcept {
    return static_cast<float>(index) * grid.step_size;
}

inline void validate(TimeGrid grid) {
    if (grid.steps_per_year == 0U
        || !(grid.step_size > 0.0f)
        || !std::isfinite(grid.step_size)) {
        throw std::invalid_argument(
            "A time grid requires a positive finite number of steps per year."
        );
    }
    const float expected =
        1.0f / static_cast<float>(grid.steps_per_year);
    if (grid.step_size != expected) {
        throw std::invalid_argument(
            "The time-grid step must equal 1 / steps_per_year."
        );
    }
}

inline std::uint32_t index(
    float time,
    TimeGrid grid,
    const char* name
) {
    validate(grid);
    if (!(time > 0.0f) || !std::isfinite(time)) {
        throw std::invalid_argument(
            std::string(name) + " must be positive and finite."
        );
    }
    const double scaled = static_cast<double>(time)
        * static_cast<double>(grid.steps_per_year);
    const double rounded = std::floor(scaled + 0.5);
    constexpr double tolerance_in_grid_steps = 1.0e-4;
    if (std::fabs(scaled - rounded) > tolerance_in_grid_steps
        || rounded < 1.0
        || rounded > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::invalid_argument(
            std::string(name) + " must lie on the configured integer time grid."
        );
    }
    return static_cast<std::uint32_t>(rounded);
}

}  // namespace ai_factory::workbench::time_grid
