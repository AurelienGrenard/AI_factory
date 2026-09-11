// Public Black-Scholes GeometricAsianOption price/delta launcher using centered spot bumping.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/markovian/black_scholes/parameters.hpp"
#include "product/geometric_asian_option/parameters.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::equity::black_scholes {

template<OptionSide Side>
void launch_black_scholes_geometric_asian_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count, const product::GeometricAsianOptionParameters* host_products,
    const product::GeometricAsianOptionParameters* device_products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset, std::size_t launch_result_count,
    float dt, std::uint32_t simulation_steps_per_day, unsigned int threads_per_block, std::size_t block_count,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices, float* device_deltas
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
