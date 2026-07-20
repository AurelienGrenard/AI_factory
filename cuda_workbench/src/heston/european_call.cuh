// Public launcher for the Heston European-call CUDA kernel.
#pragma once

#include "heston/parameters.hpp"
#include "products/european_call.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {

// Launch one specialized CUDA block for every constructed result row.
void launch_heston_european_call_cuda(
    const HestonModelParameters* device_models,
    std::size_t model_count,
    const products::EuropeanCallInput* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::heston
