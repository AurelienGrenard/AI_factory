// Shared NormalInverseGaussian American-option Longstaff-Schwartz launcher.
#pragma once

#include "common/longstaff_schwartz/launch.cuh"
#include "common/option_side.cuh"
#include "model/equity/normal_inverse_gaussian/dataset.hpp"
#include "product/american_option/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_normal_inverse_gaussian_american_option_cuda(
    const NormalInverseGaussianModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::normal_inverse_gaussian
