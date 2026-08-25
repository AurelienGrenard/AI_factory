// Market-independent conversion from integer contract days to model years.
#pragma once

#include "common/check_cuda.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::time {

struct DayFractionTimeConfiguration {
    float day_fraction;
};

inline void validate_time_configuration(
    const DayFractionTimeConfiguration& time_configuration
) {
    validate_day_fraction(time_configuration.day_fraction);
}

__device__ __forceinline__ float year_fraction(
    std::uint32_t day_count,
    const DayFractionTimeConfiguration& time_configuration
) {
    return static_cast<float>(day_count) * time_configuration.day_fraction;
}

}  // namespace ai_factory::workbench::time
