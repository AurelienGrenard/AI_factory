// Generated Public launcher for ${model_display} rate_option analytics.
#pragma once

#include "common/price_construction.cuh"

#include "common/option_side.cuh"
#include "model/fixed_income/${model}/parameters.hpp"
#include "product/rate_option/parameters.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::fixed_income::${model} {

// Launch closed-form rate_option prices across the CUDA grid.
template<OptionSide Side>
void launch_${model}_rate_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::fixed_income::${model}
