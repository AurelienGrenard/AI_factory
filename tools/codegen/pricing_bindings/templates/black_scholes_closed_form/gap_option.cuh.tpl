// Generated public launcher for closed-form Black-Scholes gap-option pricing.
#pragma once

#include "common/price_construction.cuh"

#include "common/option_side.cuh"
#include "model/equity/markovian/black_scholes/parameters.hpp"
#include "product/gap_option/parameters.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
void launch_black_scholes_gap_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GapOptionParameters* device_products,
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

}  // namespace ai_factory::workbench::model::equity::black_scholes
