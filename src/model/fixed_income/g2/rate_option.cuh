// Public launcher for G2 rate_option analytics.
#pragma once

#include "common/option_side.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/rate_option/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::g2 {

// Launch one closed-form rate_option price per CUDA thread.
template<OptionSide Side>
void launch_g2_rate_option_cuda(
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

}  // namespace ai_factory::workbench::model::g2
