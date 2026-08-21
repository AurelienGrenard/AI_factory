// Public launcher for the Merton Straddle CUDA kernel.
#pragma once

#include "model/equity/merton/dataset.hpp"
#include "product/straddle/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::merton {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
void launch_merton_straddle_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::StraddleParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::merton
