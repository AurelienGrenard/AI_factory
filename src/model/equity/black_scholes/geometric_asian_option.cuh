// Public launcher for closed-form Black-Scholes geometric-Asian pricing.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/black_scholes/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::black_scholes {

template<OptionSide Side>
void launch_black_scholes_geometric_asian_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GeometricAsianOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::black_scholes
