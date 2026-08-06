// Public launcher for the VarianceGamma fixed-strike lookback CUDA kernel.
#pragma once

#include "model/equity/variance_gamma/dataset.hpp"
#include "product/lookback_option/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::variance_gamma {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
void launch_variance_gamma_lookback_option_cuda(
    const VarianceGammaModelParameters* device_models,
    std::size_t model_count,
    const product::LookbackOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::variance_gamma
