// Public launcher for Ornstein-Uhlenbeck rate_option analytics.
#pragma once

#include "common/option_side.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/rate_option/parameters.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Launch closed-form rate_option prices across the CUDA grid.
template<OptionSide Side>
void launch_ornstein_uhlenbeck_rate_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
