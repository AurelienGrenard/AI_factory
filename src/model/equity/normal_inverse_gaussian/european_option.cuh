// Public launcher for the NormalInverseGaussian European-option CUDA kernel.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/normal_inverse_gaussian/dataset.hpp"
#include "product/european_option/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
template<OptionSide Side>
void launch_normal_inverse_gaussian_european_option_cuda(
    const NormalInverseGaussianModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
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
