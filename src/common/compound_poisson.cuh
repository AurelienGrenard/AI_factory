// Canonical Philox sources and count draw for compound-Poisson dynamics.
#pragma once

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ai_factory::workbench::compound_poisson {

// A fixed source map prevents variable Poisson or mark consumption from
// shifting the continuous driver, another event source, or a later step.
inline constexpr std::uint8_t kContinuousSource = 1U;
inline constexpr std::uint8_t kCountSource = 2U;
inline constexpr std::uint8_t kMarkSource = 3U;
inline constexpr std::uint8_t kThinningSource = 4U;
inline constexpr std::uint8_t kExtensionArrivalSource = 5U;
inline constexpr std::uint8_t kExtensionMarkSource = 6U;

template<typename UniformSequence>
__device__ __forceinline__ std::uint32_t draw_count(
    UniformSequence& uniforms,
    float mean,
    float zero_probability
) {
    constexpr float kInversionThreshold = 10.0f;
    return mean < kInversionThreshold
        ? philox::poisson_from_uniform(
            uniforms.next(), mean, zero_probability
        )
        : philox::poisson_from_uniform_sequence(uniforms, mean);
}

}  // namespace ai_factory::workbench::compound_poisson
