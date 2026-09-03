// Host validation helpers shared by sampling sources and specialized engines.
#pragma once

#include "common/sample/types.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::sample {

inline void validate_uniform_bounds(
    UniformBounds bounds,
    const char* name
) {
    if (!std::isfinite(bounds.minimum)
        || !std::isfinite(bounds.maximum)
        || bounds.minimum > bounds.maximum) {
        throw std::invalid_argument(
            std::string(name) + " bounds must be finite and ordered."
        );
    }
}

inline std::size_t sample_count(
    std::size_t parameter_count,
    std::size_t paths_per_parameter
) {
    if (parameter_count == 0U || paths_per_parameter == 0U) {
        throw std::invalid_argument(
            "Parameter and paths-per-parameter counts must be positive."
        );
    }
    if (parameter_count
        > std::numeric_limits<std::size_t>::max()
            / paths_per_parameter) {
        throw std::overflow_error("Model sample count exceeds size_t.");
    }
    return parameter_count * paths_per_parameter;
}

inline std::uint32_t calendar_step_count(
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count
) {
    if (observation_count == 0U) return 0U;
    const std::uint64_t total = initial_stub_steps
        + static_cast<std::uint64_t>(observation_count - 1U)
            * steps_per_observation;
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "The sample calendar step count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(total);
}

}  // namespace ai_factory::workbench::sample
