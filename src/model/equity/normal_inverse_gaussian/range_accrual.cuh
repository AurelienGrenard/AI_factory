// Public launcher for the NormalInverseGaussian Range Accrual CUDA kernel.
#pragma once

#include "model/equity/normal_inverse_gaussian/dataset.hpp"
#include "product/range_accrual/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
void launch_normal_inverse_gaussian_range_accrual_cuda(
    const NormalInverseGaussianModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
