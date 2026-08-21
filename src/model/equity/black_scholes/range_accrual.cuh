// Public launcher for closed-form Black-Scholes range-accrual pricing.
#pragma once

#include "model/equity/black_scholes/dataset.hpp"
#include "product/range_accrual/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::black_scholes {

void launch_black_scholes_range_accrual_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
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
