// Public launcher for closed-form Black-Scholes straddle pricing.
#pragma once

#include "model/equity/black_scholes/dataset.hpp"
#include "product/straddle/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::black_scholes {

void launch_black_scholes_straddle_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::StraddleParameters* device_products,
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

}  // namespace ai_factory::workbench::black_scholes
