// Caller-owned gradient outputs and common launch metadata.
#pragma once

#include "common/check_cuda.cuh"
#include "common/price_gradients/execution_geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::price_gradients {

enum class PricingMethod { monte_carlo, closed_form };
struct LaunchConfiguration {
    PricingMethod method = PricingMethod::monte_carlo;
    std::size_t result_offset = 0U;
    std::size_t result_count = 1U;
    std::size_t paths_per_price = 1U << 20U;
    unsigned int threads_per_block = kDefaultThreadsPerBlock;
    std::size_t block_count = 1U;
    std::uint64_t base_seed = 0U;
    unsigned int sensitivity_batch_size = kSensitivitiesPerBlock;
};
struct Outputs {
    float* prices;
    float* price_standard_errors;
    float* gradients;
    float* gradient_standard_errors;
    std::size_t price_capacity;
    std::size_t gradient_capacity;  // row-major rows * K
};

struct BufferRange { std::uintptr_t begin, end; };
inline BufferRange checked_buffer_range(const void* pointer, std::size_t count, std::size_t element_bytes) {
    validate_device_pointer(pointer, "price-gradient buffer");
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (count > (std::numeric_limits<std::uintptr_t>::max() - begin) / element_bytes)
        throw std::overflow_error("Price-gradient buffer address overflow.");
    return {begin, begin + count * element_bytes};
}

}  // namespace ai_factory::workbench::price_gradients
