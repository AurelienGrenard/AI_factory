// Public launcher for closed-form Black-Scholes asset-or-nothing pricing.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/black_scholes/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::black_scholes {

template<OptionSide Side>
void launch_black_scholes_asset_or_nothing_option_cuda(
    const BlackScholesModelParameters* device_models,
    std::size_t model_count,
    const product::AssetOrNothingOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::black_scholes
